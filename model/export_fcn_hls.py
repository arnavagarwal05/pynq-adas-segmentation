"""
Export SimpleFCN weights for HLS deployment.
Folds BN into Conv, quantizes to INT8, exports C header.

Usage:
    python export_fcn_hls.py \
        --checkpoint checkpoints_fcn_idd/best.pth \
        --test-image test_road.jpg \
        --output-dir hls_fcn/
"""

import os
import argparse
import numpy as np
import torch
import cv2
from simple_fcn import SimpleFCN


def fold_bn(conv, bn):
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    w = conv.weight.data * scale.reshape(-1, 1, 1, 1)
    b = scale * (conv.bias.data - mean) + beta
    return w, b


def quantize(tensor, bits=8):
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return np.zeros(tensor.shape, dtype=np.int8), 1.0, 0
    scale = max_val / 127.0
    q = torch.round(tensor / scale).clamp(-128, 127)
    frac = int(np.floor(-np.log2(scale))) if scale > 0 else 0
    frac = max(0, min(frac, 15))
    return q.numpy().astype(np.int8), scale, frac


def arr_to_c(name, arr, dtype="data_t"):
    shape_str = "".join(f"[{s}]" for s in arr.shape)
    lines = [f"const {dtype} {name}{shape_str} = {{"]
    flat = arr.flatten()
    for i in range(0, len(flat), 16):
        chunk = flat[i:i+16]
        vals = ", ".join(str(int(v)) for v in chunk)
        comma = "," if i + 16 < len(flat) else ""
        lines.append(f"    {vals}{comma}")
    lines.append("};")
    return "\n".join(lines)


def export_weights(model, output_path, input_frac=5):
    model.eval()
    lines = [
        "#pragma once",
        '#include "top.h"',
        "",
        "// SimpleFCN weights — auto-generated",
        "",
    ]

    frac = input_frac
    layer_info = []

    def do_conv_bn(conv, bn, prefix):
        nonlocal frac
        w, b = fold_bn(conv, bn)
        wq, ws, wf = quantize(w)
        bq, bs, bf = quantize(b)
        shift = frac + wf - frac
        lines.append(f"#define SHIFT_{prefix.upper()} {shift}")
        lines.append(arr_to_c(f"w_{prefix}", wq))
        lines.append(arr_to_c(f"b_{prefix}", bq.astype(np.int32), "acc_t"))
        lines.append("")
        layer_info.append({"name": prefix, "shift": shift, "shape": list(wq.shape), "params": wq.size})

    def do_convt(conv, prefix):
        nonlocal frac
        wq, ws, wf = quantize(conv.weight.data)
        bq, bs, bf = quantize(conv.bias.data)
        shift = frac + wf - frac
        lines.append(f"#define SHIFT_{prefix.upper()} {shift}")
        lines.append(arr_to_c(f"w_{prefix}", wq))
        lines.append(arr_to_c(f"b_{prefix}", bq.astype(np.int32), "acc_t"))
        lines.append("")
        layer_info.append({"name": prefix, "shift": shift, "shape": list(wq.shape), "params": wq.size})

    def do_conv(conv, prefix):
        nonlocal frac
        wq, ws, wf = quantize(conv.weight.data)
        bq, bs, bf = quantize(conv.bias.data)
        shift = frac + wf - frac
        lines.append(f"#define SHIFT_{prefix.upper()} {shift}")
        lines.append(arr_to_c(f"w_{prefix}", wq))
        lines.append(arr_to_c(f"b_{prefix}", bq.astype(np.int32), "acc_t"))
        lines.append("")
        layer_info.append({"name": prefix, "shift": shift, "shape": list(wq.shape), "params": wq.size})

    lines.append("// ── Encoder ──")
    do_conv_bn(model.conv1[0], model.conv1[1], "conv1")
    do_conv_bn(model.conv2[0], model.conv2[1], "conv2")
    do_conv_bn(model.conv3[0], model.conv3[1], "conv3")

    lines.append("// ── Decoder ──")
    do_convt(model.up1, "up1")
    do_conv_bn(model.dec1[0], model.dec1[1], "dec1")
    do_convt(model.up2, "up2")
    do_conv(model.head, "head")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    total = sum(l["params"] for l in layer_info)
    print(f"Wrote: {output_path}")
    print(f"Layers:")
    for l in layer_info:
        print(f"  {l['name']:10s}: shape={l['shape']}, shift={l['shift']}")
    print(f"Total params: {total:,}")
    return layer_info


def generate_test_data(model, image_path, output_dir, frac=5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))

    # test_input.h
    img_q = (img.astype(np.float32) / 255.0 * (2**frac - 1)).astype(np.int8)
    lines = ['#pragma once', '#include "top.h"', "",
             f"const data_t test_img[128][128][3] = {{"]
    for h in range(128):
        lines.append("  {")
        for w in range(128):
            r, g, b = img_q[h, w]
            comma = "," if w < 127 else ""
            lines.append(f"    {{{r}, {g}, {b}}}{comma}")
        lines.append("  }," if h < 127 else "  }")
    lines.append("};")
    path = os.path.join(output_dir, "test_input.h")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {path}")

    # ref_output.h
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = normalize(transforms.ToTensor()(img)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(t).argmax(dim=1).squeeze().numpy()

    lines = ['#pragma once', '#include "top.h"', "",
             f"const ap_uint<2> ref_mask[128][128] = {{"]
    for h in range(128):
        vals = ", ".join(str(int(pred[h, w])) for w in range(128))
        lines.append(f"  {{{vals}}}," if h < 127 else f"  {{{vals}}}")
    lines.append("};")
    path = os.path.join(output_dir, "ref_output.h")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {path}")

    names = ["Road", "Obstacle", "Structure", "Background"]
    for c in range(4):
        print(f"  {names[c]:12s}: {np.sum(pred==c):6d} px ({100*np.sum(pred==c)/pred.size:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-image", default=None)
    parser.add_argument("--output-dir", default="hls_fcn")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = SimpleFCN(num_classes=4)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {args.checkpoint} (mIoU={ckpt.get('best_miou', '?')})")

    export_weights(model, os.path.join(args.output_dir, "weights_data.h"))

    if args.test_image:
        generate_test_data(model, args.test_image, args.output_dir)

    print(f"\nDone! Copy hls_fcn/ contents to your Vitis HLS project.")


if __name__ == "__main__":
    main()

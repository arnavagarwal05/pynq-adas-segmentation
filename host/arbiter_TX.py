import cv2
import socket
import struct
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Interleave two camera feeds and stream to PYNQ-Z2")
    parser.add_argument("--pynq-ip", type=str, required=True, help="IP address of PYNQ-Z2")
    parser.add_argument("--port", type=int, default=9999, help="Port to stream on")
    parser.add_argument("--cam1", type=int, default=0, help="Camera index 1")
    parser.add_argument("--cam2", type=int, default=1, help="Camera index 2")
    parser.add_argument("--width", type=int, default=320, help="Frame width")
    parser.add_argument("--height", type=int, default=240, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    args = parser.parse_args()

    cap1 = cv2.VideoCapture(args.cam1)
    cap2 = cv2.VideoCapture(args.cam2)

    for cap, idx in [(cap1, args.cam1), (cap2, args.cam2)]:
        if not cap.isOpened():
            print(f"[ERROR] Camera index {idx} not found")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    print(f"[INFO] Both cameras opened: {args.width}x{args.height} @ {args.fps}fps")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[INFO] Connecting to PYNQ at {args.pynq_ip}:{args.port} ...")
    try:
        client_socket.connect((args.pynq_ip, args.port))
    except ConnectionRefusedError:
        print("[ERROR] Connection refused. Start the receiver on PYNQ first.")
        return
    print("[INFO] Connected! Streaming both cameras every frame...")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    frame_count = 0

    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                print("[WARN] Read failure, retrying...")
                continue

            # Send both frames back-to-back, tagged with camera ID
            for cam_id, frame in [(0, frame1), (1, frame2)]:
                frame = cv2.resize(frame, (args.width, args.height))
                _, encoded = cv2.imencode('.jpg', frame, encode_param)
                data = encoded.tobytes()
                header = struct.pack(">LB", len(data), cam_id)
                try:
                    client_socket.sendall(header + data)
                except BrokenPipeError:
                    print("[ERROR] Connection lost")
                    return

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[INFO] Sent {frame_count} pairs ({frame_count * 2} total frames)")

            # Local preview (optional, press Q to quit)
            preview = np.hstack((
                cv2.resize(frame1, (args.width, args.height)),
                cv2.resize(frame2, (args.width, args.height))
            ))
            cv2.imshow("Both Feeds (Q to quit)", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        cap1.release()
        cap2.release()
        client_socket.close()
        cv2.destroyAllWindows()
        print(f"[INFO] Done. Sent {frame_count} pairs ({frame_count * 2} total frames)")


if __name__ == "__main__":
    main()
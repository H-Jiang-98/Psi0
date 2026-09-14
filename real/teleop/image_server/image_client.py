import cv2
import zmq
import numpy as np

def start_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    server_ip = "192.168.123.164" 
    socket.connect(f"tcp://{server_ip}:5558")

    print(f"Connected to server at {server_ip}")
    print("Press 'q' to exit.")

    try:
        while True:
            socket.send(b"get")
            reply = socket.recv_multipart()

            if len(reply) < 3:
                continue

            rgb_bytes = reply[0]
            ir_bytes = reply[1]
            depth_bytes = reply[2]

            # RGB
            rgb_img = cv2.imdecode(
                np.frombuffer(rgb_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            # IR
            ir_img = cv2.imdecode(
                np.frombuffer(ir_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            # Depth (uint16 raw)
            depth_array = np.frombuffer(depth_bytes, np.uint16).reshape((480, 640))

            # Convert to 8bit
            depth_8bit = cv2.convertScaleAbs(depth_array, alpha=0.03)

            depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

            if rgb_img is not None:
                cv2.imshow("RealSense RGB", rgb_img)

            if ir_img is not None:
                cv2.imshow("RealSense IR (Left | Right)", ir_img)

            cv2.imshow("RealSense Depth", depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Client error: {e}")
    finally:
        cv2.destroyAllWindows()
        socket.close()
        context.term()

if __name__ == "__main__":
    start_client()

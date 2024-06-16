import click
import time

import cv2

import utils

# video_src = "/dev/video0"
# video_url = "https://videos.pexels.com/video-files/6345272/6345272-hd_1280_720_30fps.mp4"
# video_url = "https://videos.pexels.com/video-files/5607989/5607989-hd_1280_720_30fps.mp4"
# video_url = "https://www.pexels.com/download/video/5740605/?fps=25.0&h=1366&w=720"
# video_url = "https://www.pexels.com/download/video/6777262/?fps=25.0&h=960&w=506"

@click.command()
@click.option("--video_url", default="https://www.pexels.com/download/video/6777262/?fps=25.0&h=960&w=506")
@click.option("--model_type", type=click.Choice(["vits", "vitb", "vitl"]), default="vits")
@click.option("--use_pytorch", is_flag=True, default=False)
def main(video_url, model_type, use_pytorch):
    if use_pytorch:
        import pytorch_predictor
        model = pytorch_predictor.DepthAnythingV2Pytorch(model_type, device="cuda")
        output_video_path = "output_demo_pytorch.avi"
    else: # use openvino
        import openvino_predictor
        model = openvino_predictor.DepthAnythingV2OpenVINO(model_type, device="AUTO")
        output_video_path = "output_demo_openvino.avi"

    utils.download_video(video_url, "demo.mp4")
    video_src = "demo.mp4"

    video = cv2.VideoCapture(video_src)

    # Get the width and height of the frames in the input video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_out = cv2.VideoWriter(output_video_path, fourcc, 30, (width*2, height))

    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = video.read()
        if ret == False: break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth = model.predict(rgb_image)

        # Calculate FPS
        frame_count += 1
        if frame_count >= 10:  # Update FPS every 10 frames for stability
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            start_time = end_time
            frame_count = 0

        viz_image = cv2.hconcat([frame, depth], None)

        # Display FPS on the image
        cv2.putText(viz_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        video_out.write(viz_image)

        cv2.imshow("video", viz_image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            break

    # Release the video capture and close all OpenCV windows
    video.release()
    video_out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
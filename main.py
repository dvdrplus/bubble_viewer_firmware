import time
import os
from datetime import datetime, timedelta
from threading import Thread, Lock
from queue import Queue
from flask import Flask, jsonify
from picamera2 import Picamera2
from helpers import analyze_image
from flask import send_file


# Assuming analyze_image is available for use
# from analysis_module import analyze_image

# Initialize Flask App for Control
app = Flask(__name__)

# Base Directory (where the script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_IMAGE = os.path.join(BASE_DIR, "sample_image.jpg")

# Directory Setup
# IMAGE_DIR = "/home/pi/images"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
# ANALYSIS_DIR = "/home/pi/analysis_results"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_results")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Set capture and send intervals
CAPTURE_INTERVAL = 1  # seconds
SEND_INTERVAL = 5  # seconds ()

# Globals for managing image list and thread control
captured_images = []
analysis_queue = Queue()
send_lock = Lock()
last_send_time = datetime.now()

# Control flags for thread management
capture_running = False
analyze_running = False
send_running = False

# Initialize Camera
# camera = Picamera2()
# camera.configure(camera.create_still_configuration())


def capture_images():
    global capture_running
    while capture_running:
        # Capture Image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(IMAGE_DIR, f"image_{timestamp}.jpg")
        #camera.capture_file(image_path)

        # Add image to analysis queue
        analysis_queue.put(SAMPLE_IMAGE)  # analysis_queue.put(image_path)

        # Save image path for sending
        with send_lock:
            captured_images.append(image_path)

        # Wait for next capture
        time.sleep(CAPTURE_INTERVAL)


def analyze_images():
    global analyze_running
    while analyze_running or (not analysis_queue.empty() or capture_running):
        # Retrieve the next image path from the queue if available
        if not analysis_queue.empty():
            image_path = analysis_queue.get()
            if image_path is None:
                break  # Exit if None is received in the queue

            # Perform Image Analysis
            analysis_result = analyze_image(image_path)
            timestamp = os.path.basename(image_path).split("_")[1].split(".")[0]
            analysis_path = os.path.join(ANALYSIS_DIR, f"analysis_{timestamp}.txt")

            # Save analysis result
            with open(analysis_path, "w") as f:
                f.write(str(analysis_result))

            # Mark task as done in the queue
            analysis_queue.task_done()
        else:
            # If no images to analyze, sleep briefly to avoid excessive CPU usage
            time.sleep(0.5)


def send_images():
    global send_running, last_send_time
    while send_running:
        # Check if the send interval has passed and if there are images to send
        if (
            datetime.now() - last_send_time >= timedelta(seconds=SEND_INTERVAL)
            and captured_images
        ):
            with send_lock:
                # Iterate over the images to send
                for image_path in captured_images:
                    # Code to send image (e.g., save or send via WiFi to the tablet)
                    print(f"Sending image: {image_path}")
                    # Placeholder for sending logic (use HTTP or FTP)

                # Clear sent images
                captured_images.clear()
                last_send_time = datetime.now()

        # Wait a short time before rechecking
        time.sleep(1)


@app.route("/start_measurement", methods=["POST"])
def start_measurement():
    global capture_running, analyze_running, send_running
    if not capture_running:
        capture_running = True
        Thread(target=capture_images).start()
    if not analyze_running:
        analyze_running = True
        Thread(target=analyze_images).start()
    if not send_running:
        send_running = True
        Thread(target=send_images).start()
    return jsonify({"status": "Measurement started"})


@app.route("/stop_measurement", methods=["POST"])
def stop_measurement():
    global capture_running, analyze_running, send_running
    capture_running = False
    send_running = False
    # Keep analyze_running True until the queue is empty
    analyze_running = False
    return jsonify({"status": "Measurement stopping after analysis queue is empty"})


@app.route("/images/<image_name>")
def get_image(image_name):
    #image_path = os.path.join(IMAGE_DIR, image_name)
    image_path = SAMPLE_IMAGE
    return send_file(image_path, mimetype="image/jpeg")


def run_flask():
    app.run(host="0.0.0.0", port=5000)


# Start Flask in a separate thread
flask_thread = Thread(target=run_flask)
flask_thread.start()

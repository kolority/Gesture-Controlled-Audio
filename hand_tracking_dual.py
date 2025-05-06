import sys
sys.path.append('/usr/local/lib/python3.10/site-packages')
import cv2
import mediapipe as mp
import numpy as np
import time
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal
import threading
import queue
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
print("Opening camera...")
cap = cv2.VideoCapture(1)  # Try index 1 for external webcam

if not cap.isOpened():
    print("Error: Could not open camera. Please check your webcam connection.")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased from 640 to 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increased from 480 to 720
cap.set(cv2.CAP_PROP_FPS, 30)

# Give the camera time to warm up
print("Camera warming up...")
time.sleep(2)

print("Starting hand tracking. Press 'q' to quit.")

# Create a named window with larger size
cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Tracking', 1280, 720)  # Set initial window size

# Audio playback variables
audio_queue = queue.Queue(maxsize=32)  # Limit queue size to prevent memory issues
is_playing = True
audio_thread = None
audio_lock = threading.Lock()  # Lock for thread-safe access to audio variables
left_hand_speed = 1.0
right_hand_pitch = 1.0
left_hand_volume = 0.5  # Changed from 1.0 to 0.5 for safer default volume

# Distance scaling parameters
MIN_DISTANCE = 0.0
MAX_DISTANCE = 0.35
MIN_SPEED = 0.2    # Changed from 0.1 to 0.2 for more noticeable minimum
MAX_SPEED = 3.0    # Changed from 2.0 to 3.0 for more dramatic maximum
MIN_PITCH = 0.5    # Changed from 0.8 to 0.5 for lower pitch
MAX_PITCH = 2.0    # Changed from 1.5 to 2.0 for higher pitch
MIN_VOLUME = 0.0   # Minimum volume (mute)
MAX_VOLUME = 1.0   # Maximum volume

def audio_callback(outdata, frames, time, status):
    if status:
        print(f"Audio callback status: {status}")
    try:
        data = audio_queue.get_nowait()
        if len(data) < len(outdata):
            outdata[:len(data), 0] = data
            outdata[len(data):, 0] = 0
        else:
            outdata[:, 0] = data[:len(outdata)]
    except queue.Empty:
        outdata.fill(0)
        print("Audio buffer underrun - queue is empty")

def play_audio():
    global is_playing, left_hand_speed, right_hand_pitch, left_hand_volume
    try:
        audio_file = 'background_music.wav'
        if not os.path.exists(audio_file):
            print(f"Error: Audio file '{audio_file}' not found!")
            print("Please make sure you have converted your MP3 to WAV format using convert_audio.py")
            return
        print(f"Loading audio file: {audio_file}")
        sample_rate, audio_data = wavfile.read(audio_file)
        print(f"Audio loaded successfully. Sample rate: {sample_rate}, Shape: {audio_data.shape}")
        if len(audio_data.shape) > 1:
            print("Converting stereo to mono...")
            audio_data = audio_data.mean(axis=1)
        print("Normalizing audio data...")
        audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
        print("Starting audio stream...")
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            callback=audio_callback,
            dtype=np.float32,
            blocksize=512,
            latency='low'
        )
        stream.start()
        print("Audio stream started successfully")
        position = 0
        while is_playing:
            chunk_size = 512
            # Get current speed, pitch, and volume (thread-safe copy)
            with audio_lock:
                speed = left_hand_speed
                pitch = right_hand_pitch
                volume = left_hand_volume
            
            # Calculate how many samples to read based on speed
            read_size = int(chunk_size * speed)
            
            # Apply speed by reading more or fewer samples
            if position + read_size > len(audio_data):
                remaining = len(audio_data) - position
                chunk = audio_data[position:]
                position = 0
                # If we need more samples, read from the beginning
                if len(chunk) < read_size:
                    chunk = np.concatenate([chunk, audio_data[:read_size - len(chunk)]])
            else:
                chunk = audio_data[position:position + read_size]
                position += read_size
            
            try:
                # Resample to maintain original chunk size
                if speed != 1.0:
                    chunk = signal.resample(chunk, chunk_size)
                
                # Apply pitch change
                if pitch != 1.0:
                    chunk = signal.resample(chunk, int(len(chunk) * pitch))
                
                # Apply volume
                chunk = chunk * volume
                
                # Put the chunk in the queue, with timeout
                audio_queue.put(chunk, timeout=0.1)
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
                continue
            
            time.sleep(0.001)
        
        stream.stop()
        stream.close()
    except Exception as e:
        print(f"Error in audio playback: {e}")
        print("Please make sure:")
        print("1. You have converted your MP3 to WAV format using convert_audio.py")
        print("2. The WAV file is in the same directory as this script")
        print("3. Your system's audio output is working correctly")

# Start audio playback in a separate thread
print("Starting audio playback thread...")
audio_thread = threading.Thread(target=play_audio)
audio_thread.start()

frame_count = 0
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame, retrying...")
        time.sleep(1)
        continue
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Frame {frame_count} captured successfully")
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Reset detection flags
    left_found = False
    right_found = False
    left_hand_pos = None
    right_hand_pos = None
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_type = results.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = np.sqrt(
                (thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2
            )
            y_position = 30 + (idx * 30)
            
            # Draw line between thumb and index finger
            thumb_pixel = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_pixel = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            cv2.line(frame, thumb_pixel, index_pixel, (255, 255, 255), 2)
            
            # Store hand positions for distance calculation
            if hand_type == 'Left':
                left_found = True
                left_hand_pos = (thumb_tip.x, thumb_tip.y)
                # Map distance to volume
                norm_dist = np.clip((distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE), 0, 1)
                left_hand_volume = MIN_VOLUME + norm_dist * (MAX_VOLUME - MIN_VOLUME)
                
                # Calculate center point of the line between thumb and index
                center_x = (thumb_pixel[0] + index_pixel[0]) // 2
                center_y = (thumb_pixel[1] + index_pixel[1]) // 2
                
                # Display the volume at the center of the line
                volume_text = f"Volume: {left_hand_volume:.2f}"
                # Get text size to center it properly
                text_size = cv2.getTextSize(volume_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                
                # Draw background rectangle for better visibility
                padding = 5
                cv2.rectangle(frame, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0),
                            -1)
                
                # Draw the volume text
                cv2.putText(
                    frame,
                    volume_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # Show distance in the corner
                cv2.putText(
                    frame,
                    f"Left Hand Dist: {distance:.2f}",
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1
                )
            elif hand_type == 'Right':
                right_found = True
                right_hand_pos = (thumb_tip.x, thumb_tip.y)
                # Map distance to pitch
                norm_dist = np.clip((distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE), 0, 1)
                right_hand_pitch = MIN_PITCH + norm_dist * (MAX_PITCH - MIN_PITCH)
                
                # Calculate center point of the line between thumb and index
                center_x = (thumb_pixel[0] + index_pixel[0]) // 2
                center_y = (thumb_pixel[1] + index_pixel[1]) // 2
                
                # Display the pitch at the center of the line
                pitch_text = f"Pitch: {right_hand_pitch:.2f}x"
                # Get text size to center it properly
                text_size = cv2.getTextSize(pitch_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                
                # Draw background rectangle for better visibility
                padding = 5
                cv2.rectangle(frame, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0),
                            -1)
                
                # Draw the pitch text
                cv2.putText(
                    frame,
                    pitch_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # Show distance in the corner
                cv2.putText(
                    frame,
                    f"Right Hand Dist: {distance:.2f}",
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    1
                )
    
    # Calculate and display distance between hands if both are detected
    if left_found and right_found and left_hand_pos and right_hand_pos:
        hand_distance = np.sqrt(
            (left_hand_pos[0] - right_hand_pos[0])**2 + 
            (left_hand_pos[1] - right_hand_pos[1])**2
        )
        # Draw a line between hands
        left_pixel = (int(left_hand_pos[0] * frame.shape[1]), int(left_hand_pos[1] * frame.shape[0]))
        right_pixel = (int(right_hand_pos[0] * frame.shape[1]), int(right_hand_pos[1] * frame.shape[0]))
        cv2.line(frame, left_pixel, right_pixel, (255, 255, 255), 2)
        
        # Calculate center point of the line
        center_x = (left_pixel[0] + right_pixel[0]) // 2
        center_y = (left_pixel[1] + right_pixel[1]) // 2
        
        # Map hand distance to speed with adjusted scaling
        # Scale the distance to be more sensitive to smaller movements
        scaled_distance = hand_distance * 2.0  # Multiply by 2 to make it more sensitive
        norm_dist = np.clip(scaled_distance, 0, 1)
        # Use cubic mapping for even more dramatic effect
        norm_dist = norm_dist * norm_dist * norm_dist
        left_hand_speed = MIN_SPEED + norm_dist * (MAX_SPEED - MIN_SPEED)
        
        # Display the speed at the center of the line
        speed_text = f"Speed: {left_hand_speed:.2f}x"
        # Get text size to center it properly
        text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        
        # Draw background rectangle for better visibility
        padding = 5
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     (0, 0, 0),
                     -1)
        
        # Draw the speed text
        cv2.putText(
            frame,
            speed_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
    
    with audio_lock:
        if not left_found or not right_found:
            left_hand_speed = 1.0
        if not right_found:
            right_hand_pitch = 1.0
        if not left_found:
            left_hand_volume = 1.0
    
    # Display current speed on the frame
    cv2.putText(
        frame,
        f"Speed: {left_hand_speed:.2f}x",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
print("Closing camera...")
is_playing = False
if audio_thread:
    audio_thread.join()
cap.release()
cv2.destroyAllWindows() 
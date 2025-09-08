import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from scipy import signal
from collections import deque
import threading

# ========================= CONFIGURATION CONSTANTS =========================

# ROI (Region of Interest) parameters for forehead detection
ROI_X1 = 0.4      # Left boundary (40% of face width)
ROI_X2 = 0.6      # Right boundary (60% of face width)  
ROI_Y1 = 0.1      # Upper boundary (10% of face height)
ROI_Y2 = 0.25     # Lower boundary (25% of face height)

# Camera settings
CAMERA_INDEX = 0          # Default camera (0 = first camera)
TARGET_FPS = 30           # Target frames per second
CAMERA_BACKEND = cv2.CAP_DSHOW  # Windows camera backend

# Signal processing parameters
BUFFER_SIZE = 250         # Signal buffer size (~8 seconds at 30 FPS)
MIN_SIGNAL_LENGTH = 120   # Minimum samples for calculation (4 seconds)
FACE_DETECTION_INTERVAL = 15  # Face detection every N frames

# Heart rate parameters
HR_MIN_FREQ = 0.75        # Minimum heart rate frequency (45 BPM)
HR_MAX_FREQ = 2.5         # Maximum heart rate frequency (150 BPM)
HR_MIN_BPM = 45           # Minimum acceptable BPM
HR_MAX_BPM = 150          # Maximum acceptable BPM
HR_BUFFER_SIZE = 5        # Heart rate smoothing buffer

# Breathing rate parameters  
BR_MIN_FREQ = 0.1         # Minimum breathing frequency (6 breaths/min)
BR_MAX_FREQ = 0.4         # Maximum breathing frequency (24 breaths/min)
BR_MIN_RATE = 6           # Minimum acceptable breathing rate
BR_MAX_RATE = 30          # Maximum acceptable breathing rate
BR_BUFFER_SIZE = 5        # Breathing rate smoothing buffer
BR_MIN_SIGNAL_LENGTH = 150  # Minimum samples for breathing (5 seconds)

# Signal quality parameters
MIN_SIGNAL_QUALITY = 0.3  # Minimum quality for calculations
MOTION_THRESHOLD = 25     # Motion detection threshold
PLOT_UPDATE_INTERVAL = 5  # Update plot every N frames

# Filter parameters
FILTER_ORDER = 3          # Butterworth filter order
NORMALIZATION_WINDOW = 10 # Window for signal normalization

# UI parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
INFO_PANEL_COLOR = (0, 0, 0)      # Black background for info
INFO_PANEL_ALPHA = 0.7             # Transparency
ROI_COLOR = (0, 255, 0)           # Green ROI rectangle
TEXT_COLOR = (0, 255, 0)          # Green text
QUALITY_COLORS = {
    'good': (0, 255, 0),      # Green for good quality
    'medium': (0, 255, 255),  # Yellow for medium quality  
    'poor': (0, 0, 255)       # Red for poor quality
}

class VitalSignsMonitor:
    def __init__(self):
        """Initialize the vital signs monitoring system"""
        # Use configuration constants
        self.x1, self.x2 = ROI_X1, ROI_X2
        self.y1, self.y2 = ROI_Y1, ROI_Y2
        
        # Initialize face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Initialize camera with configuration
        self.cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.step = int(1000/self.fps)
        
        # Signal processing with configuration
        self.green_values = deque(maxlen=BUFFER_SIZE)
        self.timestamps = deque(maxlen=BUFFER_SIZE)
        
        # Smoothing buffers with configuration
        self.heart_rate_buffer = deque(maxlen=HR_BUFFER_SIZE)
        self.breath_rate_buffer = deque(maxlen=BR_BUFFER_SIZE)
        
        # UI configuration
        self.font = FONT
        
        # Control flags
        self.running = True
        self.plot_enabled = True
        self.insights_enabled = True  # Show measurement insights
        
        print(f"Camera FPS: {self.fps}")
        print(f"Configuration loaded:")
        print(f"- ROI: ({ROI_X1}, {ROI_Y1}) to ({ROI_X2}, {ROI_Y2})")
        print(f"- Heart Rate Range: {HR_MIN_BPM}-{HR_MAX_BPM} BPM")
        print(f"- Breathing Range: {BR_MIN_RATE}-{BR_MAX_RATE} BrPM")
        
    def get_face_roi(self, img):
        """Detect face and return ROI coordinates for forehead"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
        
        if len(faces) > 0:
            face = faces[0]
            # Calculate forehead ROI
            roi_x1 = face[0] + int(self.x1 * face[2])
            roi_y1 = face[1] + int(self.y1 * face[3])
            roi_x2 = face[0] + int(self.x2 * face[2])
            roi_y2 = face[1] + int(self.y2 * face[3])
            
            return [roi_x1, roi_y1, roi_x2, roi_y2]
        return [0, 0, 0, 0]
    
    def get_color_average(self, frame, color_channel):
        """Calculate average color value for specific channel"""
        if frame.size == 0:
            return 0
        return np.mean(frame[:, :, color_channel])
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=FILTER_ORDER):
        """Apply bandpass filter to isolate specific frequencies"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if len(data) < 4 * order:
            return data
            
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            return signal.filtfilt(b, a, data)
        except:
            return data
    
    def calculate_heart_rate(self, filtered_signal, fs):
        """Calculate heart rate from filtered signal using FFT with configuration"""
        if len(filtered_signal) < MIN_SIGNAL_LENGTH // 1.5:  # Need at least 3 seconds
            return 0
            
        # Apply window to reduce spectral leakage
        windowed_signal = filtered_signal * signal.windows.hann(len(filtered_signal))
        
        # Apply FFT
        fft = np.fft.fft(windowed_signal)
        freqs = np.fft.fftfreq(len(windowed_signal), 1/fs)
        
        # Heart rate range using configuration
        hr_range = (freqs >= HR_MIN_FREQ) & (freqs <= HR_MAX_FREQ)
        
        if not np.any(hr_range):
            return 0
            
        # Find dominant frequency
        hr_freqs = freqs[hr_range]
        hr_fft = np.abs(fft[hr_range])
        
        if len(hr_fft) == 0:
            return 0
            
        # Apply smoothing to FFT
        if len(hr_fft) > 5:
            hr_fft = signal.savgol_filter(hr_fft, 5, 2)
        
        dominant_freq = hr_freqs[np.argmax(hr_fft)]
        heart_rate = dominant_freq * 60  # Convert to BPM
        
        # Validation using configuration constants
        if heart_rate < HR_MIN_BPM or heart_rate > HR_MAX_BPM:
            return 0
            
        return heart_rate
    
    def calculate_breath_rate(self, raw_signal, fs):
        """Calculate breathing rate using configuration parameters"""
        if len(raw_signal) < BR_MIN_SIGNAL_LENGTH:
            return 0
            
        # Filter using configuration
        filtered = self.bandpass_filter(raw_signal, BR_MIN_FREQ, BR_MAX_FREQ, fs)
        
        # Apply window
        windowed_signal = filtered * signal.windows.hann(len(filtered))
        
        # Apply FFT
        fft = np.fft.fft(windowed_signal)
        freqs = np.fft.fftfreq(len(windowed_signal), 1/fs)
        
        # Breathing rate range using configuration
        br_range = (freqs >= BR_MIN_FREQ) & (freqs <= BR_MAX_FREQ)
        
        if not np.any(br_range):
            return 0
            
        br_freqs = freqs[br_range]
        br_fft = np.abs(fft[br_range])
        
        if len(br_fft) == 0:
            return 0
            
        dominant_freq = br_freqs[np.argmax(br_fft)]
        breath_rate = dominant_freq * 60
        
        return breath_rate
    
    def draw_info(self, frame, bbox, heart_rate, breath_rate, signal_quality):
        """Draw information overlay using configuration constants"""
        # Draw ROI rectangle using configuration
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), ROI_COLOR, 2)
        
        # Draw info panel with configuration
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (450, 160), INFO_PANEL_COLOR, -1)
        cv2.addWeighted(overlay, INFO_PANEL_ALPHA, frame, 1-INFO_PANEL_ALPHA, 0, frame)
        
        # Draw info text using configuration
        info_y = 30
        cv2.putText(frame, f"Heart Rate: {heart_rate:.1f} BPM", 
                   (10, info_y), self.font, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        info_y += 30
        cv2.putText(frame, f"Breath Rate: {breath_rate:.1f} BrPM", 
                   (10, info_y), self.font, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        info_y += 30
        # Quality color based on thresholds
        if signal_quality > 0.7:
            quality_color = QUALITY_COLORS['good']
        elif signal_quality > 0.4:
            quality_color = QUALITY_COLORS['medium']
        else:
            quality_color = QUALITY_COLORS['poor']
            
        cv2.putText(frame, f"Signal Quality: {signal_quality:.2f}", 
                   (10, info_y), self.font, FONT_SCALE, quality_color, FONT_THICKNESS)
        
        info_y += 30
        cv2.putText(frame, "Controls: ESC/Q-Exit | SPACE-Reset | P-Plot", 
                   (10, info_y), self.font, 0.5, (255, 255, 255), 1)
        
        info_y += 20
        cv2.putText(frame, "C-Config | H-Help | I-Insights | ENTER-Exit", 
                   (10, info_y), self.font, 0.5, (255, 255, 255), 1)
    
    def calculate_signal_quality(self, signal_data):
        """Calculate signal quality based on signal stability"""
        if len(signal_data) < 30:
            return 0
            
        # Calculate signal-to-noise ratio
        signal_mean = np.mean(signal_data)
        signal_std = np.std(signal_data)
        
        if signal_mean == 0:
            return 0
            
        # Normalize quality score
        snr = signal_mean / (signal_std + 1e-6)
        quality = min(1.0, snr / 10.0)
        return max(0, quality)
    
    def safe_plot_update(self, ax1, ax2):
        """Safely update plots in a separate thread"""
        try:
            if not self.plot_enabled or len(self.green_values) < 10:
                return
                
            # Raw signal plot
            ax1.clear()
            ax1.plot(list(self.green_values)[-200:], 'g-', linewidth=1)
            ax1.set_title('Raw PPG Signal (Green Channel)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True)
            
            # Filtered signal plot
            if len(self.green_values) >= 90:
                ax2.clear()
                signal_array = np.array(self.green_values)
                filtered_signal = self.bandpass_filter(signal_array[-150:], 0.75, 2.5, self.fps)
                ax2.plot(filtered_signal, 'r-', linewidth=1)
                ax2.set_title('Filtered Signal (Heart Rate Band)')
                ax2.set_ylabel('Amplitude')
                ax2.set_xlabel('Samples')
                ax2.grid(True)
            
            plt.tight_layout()
            plt.pause(0.001)
            
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def print_configuration(self):
        """Print current configuration parameters"""
        print("\n=== CURRENT CONFIGURATION ===")
        print(f"ROI Parameters:")
        print(f"  X1: {ROI_X1} ({ROI_X1*100:.0f}% from left)")
        print(f"  X2: {ROI_X2} ({ROI_X2*100:.0f}% from left)")
        print(f"  Y1: {ROI_Y1} ({ROI_Y1*100:.0f}% from top)")
        print(f"  Y2: {ROI_Y2} ({ROI_Y2*100:.0f}% from top)")
        print(f"Heart Rate:")
        print(f"  Frequency Range: {HR_MIN_FREQ:.2f}-{HR_MAX_FREQ:.2f} Hz")
        print(f"  BPM Range: {HR_MIN_BPM}-{HR_MAX_BPM}")
        print(f"  Buffer Size: {HR_BUFFER_SIZE}")
        print(f"Breathing Rate:")
        print(f"  Frequency Range: {BR_MIN_FREQ:.2f}-{BR_MAX_FREQ:.2f} Hz")
        print(f"  Rate Range: {BR_MIN_RATE}-{BR_MAX_RATE}")
        print(f"  Buffer Size: {BR_BUFFER_SIZE}")
        print(f"Signal Processing:")
        print(f"  Buffer Size: {BUFFER_SIZE} samples")
        print(f"  Min Signal Quality: {MIN_SIGNAL_QUALITY}")
        print(f"  Motion Threshold: {MOTION_THRESHOLD}")
        print("==============================\n")
    
    def print_measurement_factors(self):
        """Print factors that affect pulse and breathing measurements"""
        print("\n=== FACTORS AFFECTING MEASUREMENTS ===")
        print("HEART RATE - Higher readings may be caused by:")
        print("  Physiological: Exercise, stress, caffeine, fever")
        print("  Technical: Poor lighting, movement, large ROI")
        print("HEART RATE - Lower readings may be caused by:")
        print("  Physiological: Rest, fitness, medications, cold")
        print("  Technical: Very stable position, small ROI")
        print("BREATHING - Higher readings may be caused by:")
        print("  Physiological: Activity, anxiety, illness, heat")
        print("  Technical: Head movements, external vibrations")
        print("BREATHING - Lower readings may be caused by:")
        print("  Physiological: Relaxation, controlled breathing")
        print("  Technical: Shallow breathing, breath holding")
        print("TIPS FOR ACCURACY:")
        print("  - Stay very still during measurement")
        print("  - Ensure good, stable lighting")
        print("  - Wait 5-10 minutes after exercise")
        print("  - Aim for signal quality > 0.6")
        print("  - Allow 10-15 seconds for stabilization")
    def analyze_signal_characteristics(self, heart_rate, breath_rate, signal_quality, motion_level):
        """Analyze current measurements and provide insights"""
        insights = []
        
        # Heart rate analysis
        if heart_rate > 0:
            if heart_rate > 100:
                insights.append("High HR: Check for stress/activity/caffeine")
            elif heart_rate < 60:
                insights.append("Low HR: May indicate fitness or relaxation")
            
            if heart_rate > HR_MAX_BPM * 0.9:
                insights.append("Very high HR: Check measurement conditions")
            elif heart_rate < HR_MIN_BPM * 1.1:
                insights.append("Very low HR: Verify signal quality")
        
        # Breathing rate analysis  
        if breath_rate > 0:
            if breath_rate > 20:
                insights.append("Fast breathing: Check for anxiety/activity")
            elif breath_rate < 10:
                insights.append("Slow breathing: May indicate relaxation")
        
        # Signal quality analysis
        if signal_quality < 0.5:
            insights.append("Low quality: Improve lighting/reduce movement")
        elif signal_quality > 0.8:
            insights.append("Excellent signal quality")
            
        # Motion analysis
        if motion_level > MOTION_THRESHOLD * 1.5:
            insights.append("High movement detected: Stay more still")
        
        return insights
    
    def draw_insights(self, frame, insights):
        """Draw measurement insights on frame"""
        if not insights:
            return
            
        # Draw insights panel
        panel_height = len(insights) * 25 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1] - 400, 10), 
                     (frame.shape[1] - 10, 10 + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw insights text
        y_pos = 30
        cv2.putText(frame, "Measurement Insights:", 
                   (frame.shape[1] - 390, y_pos), 
                   self.font, 0.6, (255, 255, 0), 1)
        
        for insight in insights[:6]:  # Show max 6 insights
            y_pos += 25
            cv2.putText(frame, f"• {insight}", 
                       (frame.shape[1] - 390, y_pos), 
                       self.font, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Main monitoring loop"""
        # Setup matplotlib - non-blocking
        if self.plot_enabled:
            try:
                plt.ion()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                fig.suptitle('Vital Signs Monitor')
                plt.show(block=False)
            except Exception as e:
                print(f"Plot initialization failed: {e}")
                self.plot_enabled = False
                ax1, ax2 = None, None
        else:
            ax1, ax2 = None, None
        
        # Initialize variables
        frame_count = 0
        bbox = [100, 100, 200, 150]
        previous_frame = None
        
        # Check camera
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read from camera")
            return
        
        print("Starting vital signs monitoring...")
        print("Position yourself in front of the camera with good lighting")
        print("Stay still for accurate measurements")
        print("Controls: ESC/Q - Exit | SPACE - Reset | P - Toggle Plot | ENTER - Force Exit")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Face detection using configuration interval
                if frame_count % FACE_DETECTION_INTERVAL == 0 or previous_frame is None:
                    new_roi = self.get_face_roi(frame)
                    if new_roi[2] > new_roi[0]:
                        bbox = new_roi
                
                # Motion detection using configuration threshold
                motion_level = 0
                if previous_frame is not None:
                    diff = cv2.absdiff(frame, previous_frame)
                    motion_level = np.mean(diff)
                    if motion_level > MOTION_THRESHOLD:
                        new_roi = self.get_face_roi(frame)
                        if new_roi[2] > new_roi[0]:
                            bbox = new_roi
                
                # Extract ROI and calculate green channel average
                roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if roi.size > 0:
                    # Use normalized green channel with configuration
                    green_avg = self.get_color_average(roi, 1)
                    # Apply normalization using configuration window
                    if len(self.green_values) > 0:
                        recent_mean = np.mean(list(self.green_values)[-NORMALIZATION_WINDOW:])
                        green_avg = green_avg / (recent_mean + 1e-6)
                    
                    self.green_values.append(green_avg)
                    self.timestamps.append(current_time)
                
                # Calculate vital signs using configuration parameters
                heart_rate = 0
                breath_rate = 0
                signal_quality = 0
                
                if len(self.green_values) >= MIN_SIGNAL_LENGTH:
                    signal_array = np.array(self.green_values)
                    
                    # Calculate signal quality
                    signal_quality = self.calculate_signal_quality(
                        signal_array[-MIN_SIGNAL_LENGTH//2:]
                    )
                    
                    # Only calculate if signal quality meets minimum threshold
                    if signal_quality > MIN_SIGNAL_QUALITY:
                        # Calculate heart rate using configuration
                        hr_filtered = self.bandpass_filter(
                            signal_array, HR_MIN_FREQ, HR_MAX_FREQ, self.fps
                        )
                        heart_rate = self.calculate_heart_rate(hr_filtered, self.fps)
                        
                        # Calculate breathing rate
                        breath_rate = self.calculate_breath_rate(signal_array, self.fps)
                        
                        # Apply smoothing with validation using configuration
                        if HR_MIN_BPM <= heart_rate <= HR_MAX_BPM:
                            self.heart_rate_buffer.append(heart_rate)
                            if len(self.heart_rate_buffer) >= 3:
                                heart_rate = np.median(self.heart_rate_buffer)
                        
                        if BR_MIN_RATE <= breath_rate <= BR_MAX_RATE:
                            self.breath_rate_buffer.append(breath_rate)
                            if len(self.breath_rate_buffer) >= 3:
                                breath_rate = np.median(self.breath_rate_buffer)
                
                # Draw information on frame
                self.draw_info(frame, bbox, heart_rate, breath_rate, signal_quality)
                
                # Draw insights if enabled
                if self.insights_enabled and len(self.green_values) >= MIN_SIGNAL_LENGTH:
                    insights = self.analyze_signal_characteristics(
                        heart_rate, breath_rate, signal_quality, motion_level
                    )
                    self.draw_insights(frame, insights)
                
                # Update plots using configuration interval
                if self.plot_enabled and ax1 is not None and frame_count % PLOT_UPDATE_INTERVAL == 0:
                    self.safe_plot_update(ax1, ax2)
                
                # Display video
                cv2.imshow("Vital Signs Monitor", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or Q
                    print("Exiting...")
                    break
                elif key == 13:  # ENTER - Force exit
                    print("Force exit...")
                    self.running = False
                    break
                elif key == ord(' '):  # SPACE - reset
                    self.green_values.clear()
                    self.heart_rate_buffer.clear()
                    self.breath_rate_buffer.clear()
                    print("Signal buffers reset")
                elif key == ord('p'):  # P - toggle plot
                    self.plot_enabled = not self.plot_enabled
                    print(f"Plot {'enabled' if self.plot_enabled else 'disabled'}")
                elif key == ord('c'):  # C - print current configuration
                    self.print_configuration()
                elif key == ord('h'):  # H - help with measurement factors
                    self.print_measurement_factors()
                elif key == ord('i'):  # I - toggle insights
                    self.insights_enabled = not self.insights_enabled
                    print(f"Insights {'enabled' if self.insights_enabled else 'disabled'}")
                
                # Check if window is closed
                try:
                    if cv2.getWindowProperty('Vital Signs Monitor', cv2.WND_PROP_AUTOSIZE) < 1:
                        break
                except:
                    break
                
                previous_frame = frame.copy()
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        
        finally:
            # Cleanup
            self.running = False
            try:
                self.cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                if self.plot_enabled:
                    plt.close('all')
            except:
                pass
            print("Application closed successfully")

def main():
    """Main function"""
    print("=== Vital Signs Monitor ===")
    print("Real-time contactless heart rate and breathing monitoring")
    print("Make sure you have good lighting and stay still during measurement")
    print()
    
    try:
        monitor = VitalSignsMonitor()
        monitor.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and not being used by another application")

if __name__ == "__main__":
    main()

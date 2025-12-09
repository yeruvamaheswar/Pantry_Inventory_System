import cv2
import numpy as np
from pyzbar.pyzbar import decode
import time
import json
import os
import sys

# --- CONFIGURATION & CONSTANTS ---
CAMERA_ID = 0  # 0 is usually the default USB webcam
ZONE_FILE = "zones.json" # File where we will save our zone settings

# Resolution settings
CAPTURE_WIDTH = 3840  # 4K width
CAPTURE_HEIGHT = 2160  # 4K height
DISPLAY_WIDTH = 1920  # 1080p width
DISPLAY_HEIGHT = 1080  # 1080p height
SCALE_FACTOR = DISPLAY_WIDTH / CAPTURE_WIDTH  # 0.5

# Colors for drawing on the screen (Blue, Green, Red)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

def beep():
    """Plays a system beep."""
    sys.stdout.write('\a')
    sys.stdout.flush()

def scale_to_capture(x, y):
    """Scale display coordinates (1080p) to capture coordinates (4K)."""
    return int(x / SCALE_FACTOR), int(y / SCALE_FACTOR)

def scale_to_display(x, y):
    """Scale capture coordinates (4K) to display coordinates (1080p)."""
    return int(x * SCALE_FACTOR), int(y * SCALE_FACTOR)

def scale_rect_to_capture(rect):
    """Scale a rectangle from display to capture coordinates."""
    x, y, w, h = rect
    return (int(x / SCALE_FACTOR), int(y / SCALE_FACTOR), 
            int(w / SCALE_FACTOR), int(h / SCALE_FACTOR))

def scale_rect_to_display(rect):
    """Scale a rectangle from capture to display coordinates."""
    x, y, w, h = rect
    return (int(x * SCALE_FACTOR), int(y * SCALE_FACTOR), 
            int(w * SCALE_FACTOR), int(h * SCALE_FACTOR))

def enhance_image_for_scanning(image):
    """
    Makes the image black & white and sharper to help the scanner.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    return sharpened

    
    return sharpened

class ZoneManager:
    """
    This class handles saving and loading the zone definitions(boxes) from a file.

    It's like a librarian for your shelf setup.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.zones = self.load_zones()

    def load_zones(self):
        """Attempts to read zones from the JSON file. Returns default if file missing."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading zones: {e}")
        
        # Default empty structure if no file found
        return {"barcode_zone": None, "shelves": []}

    def save_zones(self):
        """Writes the current zones to the JSON file."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.zones, f, indent=4)
            print("Zones saved successfully!")
        except Exception as e:
                print(f"Error saving zones: {e}")

    def add_shelf(self, rect):
        """Adds a new shelf box to our list."""
        # rect is (x, y, width, height)
        self.zones["shelves"].append(rect)

    def set_barcode_zone(self, rect):
        """Sets the main scanning area."""
        self.zones["barcode_zone"] = rect

    def clear_shelves(self):
        """Deletes all shelf definitions."""
        self.zones["shelves"] = []

    def delete_barcode_zone(self):
        self.zones["barcode_zone"] = None

    def delete_shelf(self, index):
        if 0 <= index < len(self.zones["shelves"]):
            self.zones["shelves"].pop(index)

    def get_clicked_zone(self, x, y):
        """
        Returns ('barcode', None) or ('shelf', index) if (x,y) is inside a zone.
        x, y are in DISPLAY coordinates (1080p).
        Zones are stored in CAPTURE coordinates (4K).
        Returns None if nothing clicked.
        """
        # Check Barcode Zone
        bz_4k = self.zones.get("barcode_zone")
        if bz_4k:
            # Scale to display coordinates for collision check
            (bx, by, bw, bh) = scale_rect_to_display(bz_4k)
            # Normalize rectangle coordinates for collision detection if width/height are negative
            if bw < 0: bx, bw = bx + bw, -bw
            if bh < 0: by, bh = by + bh, -bh
            if (bx < x < bx + bw) and (by < y < by + bh):
                return ('barcode', None)
        
        # Check Shelves
        for i, sh_4k in enumerate(self.zones.get("shelves", [])):
            # Scale to display coordinates for collision check
            (sx, sy, sw, sh_h) = scale_rect_to_display(sh_4k)
            # Normalize rectangle coordinates for collision detection
            if sw < 0: sx, sw = sx + sw, -sw
            if sh_h < 0: sy, sh_h = sy + sh_h, -sh_h
            if (sx < x < sx + sw) and (sy < y < sy + sh_h):
                return ('shelf', i)
        
        return None

# Global variables for mouse interaction (needed because of how OpenCV handles mouse clicks)
drawing = False # True if mouse is pressed down
ix, iy = -1, -1 # Initial x, y coordinates where you clicked
current_rect = None # The rectangle currently being drawn
selected_zone = None # ('barcode', None) or ('shelf', index)

# Need to inject zm into globals for the callback to work simply
zm = None

def draw_rectangle(event, x, y, flags, param):
    """
    This function discovers what the mouse is doing.
    Mouse events are in DISPLAY coordinates (1080p).
    We scale to CAPTURE coordinates (4K) for storage.
    """
    global ix, iy, drawing, current_rect, selected_zone, zm

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if we clicked an existing zone (zones are in display coords for rendering)
        clicked = zm.get_clicked_zone(x, y)
        if clicked:
            selected_zone = clicked
            print(f"Selected: {clicked}")
            return

        # Start drawing (display coordinates)
        drawing = True
        ix, iy = x, y
        current_rect = (x, y, 0, 0)
        selected_zone = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            width = x - ix
            height = y - iy
            current_rect = (ix, iy, width, height)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            width = x - ix
            height = y - iy
            current_rect = (ix, iy, width, height)

def main():
    global current_rect, zm, selected_zone

    # Initialize the camera at 4K
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    # Load our Zone Helper (zones stored in CAPTURE coordinates - 4K)
    zm = ZoneManager(ZONE_FILE)

    # Setup the mouse listener
    cv2.namedWindow('Pantry Eye')
    cv2.setMouseCallback('Pantry Eye', draw_rectangle)

    # State variables
    mode = "RUNNING" 
    setup_step = "BARCODE" 
    
    # Tracking variables
    tracker = None
    tracking_item_name = None
    
    # Placement Logic Variables
    candidate_shelf = None
    candidate_lock_time = 0
    last_center = None
    
    print("--- Pantry Eye Active ---")
    print("Press 's' to enter Setup Mode to draw/edit zones.")
    print("Press 'q' to Quit.")
    
    while True:
        ret, frame_4k = cap.read()
        if not ret: break

        # Downscale to 1080p for display and tracking
        frame_1080p = cv2.resize(frame_4k, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        display_frame = frame_1080p.copy()
        key = cv2.waitKey(1) & 0xFF

        # --- LOGIC SWITCH: SETUP vs RUNNING ---
        if mode == "SETUP":
            cv2.putText(display_frame, f"SETUP MODE: Draw {setup_step} Zone", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)
            cv2.putText(display_frame, "Click to Select | DEL to Delete | SPACE to Confirm Box", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
            cv2.putText(display_frame, "Press 'Enter' to Save & Exit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 1)

            # Show shelf count
            shelf_count = len(zm.zones.get("shelves", []))
            cv2.putText(display_frame, f"Shelves Added: {shelf_count}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLUE, 1)

            # Draw current dragging box
            if current_rect:
                (x, y, w, h) = current_rect
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), COLOR_YELLOW, 2)

            # --- KEYBOARD COMMANDS ---
            if key == 13: 
                mode = "RUNNING"
                zm.save_zones()
                current_rect = None
                selected_zone = None
                print("Setup saved.")

            elif key == ord(' '):
                if current_rect:
                    # Scale from display (1080p) to capture (4K) before saving
                    rect_4k = scale_rect_to_capture(current_rect)
                    if setup_step == "BARCODE":
                        zm.set_barcode_zone(rect_4k)
                        print("Barcode zone set.")
                        setup_step = "SHELF"
                    else:
                        zm.add_shelf(rect_4k)
                        print("Shelf added.")
                    current_rect = None

            elif key == 127 or key == 8:
                if selected_zone:
                    ztype, zidx = selected_zone
                    if ztype == 'barcode':
                        zm.delete_barcode_zone()
                        print("Deleted Barcode Zone")
                        setup_step = "BARCODE"
                    elif ztype == 'shelf':
                        zm.delete_shelf(zidx)
                        print(f"Deleted Shelf {zidx+1}")
                    selected_zone = None

            elif key == ord('u'):
                if len(zm.zones["shelves"]) > 0:
                    zm.zones["shelves"].pop()
                    print("Undid last shelf.")
                    selected_zone = None

            # --- VISUALIZE ZONES ---
            if selected_zone:
                ztype, zidx = selected_zone
                rect = None
                if ztype == 'barcode':
                    rect = zm.zones.get("barcode_zone")
                else:
                    shelves = zm.zones.get("shelves", [])
                    if zidx is not None and 0 <= zidx < len(shelves):
                        rect = shelves[zidx]
                if rect:
                    # Scale from 4K to 1080p for display
                    (rx, ry, rw, rh) = scale_rect_to_display(rect)
                    cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), COLOR_WHITE, 3)

            bz = zm.zones.get("barcode_zone")
            if bz:
                # Scale from 4K to 1080p for display
                (bx, by, bw, bh) = scale_rect_to_display(bz)
                color = COLOR_GREEN
                if selected_zone and selected_zone[0] == 'barcode': color = COLOR_WHITE
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
                cv2.putText(display_frame, "Barcode Zone", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            for i, sh in enumerate(zm.zones.get("shelves", [])):
                # Scale from 4K to 1080p for display
                (sx, sy, sw, sh_h) = scale_rect_to_display(sh)
                color = COLOR_BLUE
                if selected_zone and selected_zone == ('shelf', i): color = COLOR_WHITE
                cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh_h), color, 2)
                cv2.putText(display_frame, f"Shelf {i+1}", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        elif mode == "RUNNING":
            # 1. DRAW ZONES (zones stored in 4K, scale to 1080p for display)
            bg_zone_4k = zm.zones.get("barcode_zone")
            bg_zone_display = None
            if bg_zone_4k:
                bg_zone_display = scale_rect_to_display(bg_zone_4k)
                (bx, by, bw, bh) = bg_zone_display
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), COLOR_GREEN, 2)
                # Extract scan area from 4K frame
                (bx_4k, by_4k, bw_4k, bh_4k) = bg_zone_4k
                if bw_4k > 0 and bh_4k > 0:
                   scan_area = frame_4k[by_4k:by_4k+bh_4k, bx_4k:bx_4k+bw_4k]
                else:
                   scan_area = frame_4k
            else:
                scan_area = frame_4k

            for i, sh_4k in enumerate(zm.zones.get("shelves", [])):
                # Scale to 1080p for display
                (sx, sy, sw, sh_h) = scale_rect_to_display(sh_4k)
                # Highlight Candidate Shelf in Yellow
                if candidate_shelf == f"Shelf {i+1}":
                    cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh_h), COLOR_YELLOW, 3)
                    cv2.putText(display_frame, f"Shelf {i+1} (CANDIDATE)", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)
                else:
                    cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh_h), COLOR_BLUE, 2)
                    cv2.putText(display_frame, f"Shelf {i+1}", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 1)

            # 2. SCANNING LOGIC (Only if NOT tracking)
            if tracker is None:
                # specific "Scanning" text
                if bg_zone_display:
                    (bx, by, bw, bh) = bg_zone_display
                    cv2.putText(display_frame, "SEARCHING...", (bx, by + bh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
                
                # Enhance the image for better detection
                enhanced_area = enhance_image_for_scanning(scan_area)
                barcodes = decode(enhanced_area)
                
                for barcode in barcodes:
                    code_data = barcode.data.decode('utf-8')
                    current_item = code_data
                    
                    # --- SUCCESS! ---
                    beep()
                    print(f"✅ SCANNED: {current_item} - Starting Tracker...")
                    tracking_item_name = current_item
                    
                    # Initialize Tracker on 1080p frame
                    tracker = cv2.TrackerCSRT_create()
                    (bx_local, by_local, bw_local, bh_local) = barcode.rect
                    
                    # Convert to GLOBAL coordinates (in 4K space)
                    if bg_zone_4k:
                         (gz_x, gz_y, _, _) = bg_zone_4k
                         global_x_4k = bx_local + gz_x
                         global_y_4k = by_local + gz_y
                    else:
                         global_x_4k = bx_local
                         global_y_4k = by_local
                    
                    # Scale to 1080p for tracking
                    global_x, global_y = scale_to_display(global_x_4k, global_y_4k)
                    bw_1080p = int(bw_local * SCALE_FACTOR)
                    bh_1080p = int(bh_local * SCALE_FACTOR)
                    
                    # Context Expansion
                    pad_w = bw_1080p * 1.5
                    pad_h = bh_1080p * 1.5
                    start_x = int(global_x - pad_w)
                    start_y = int(global_y - pad_h)
                    start_w = int(bw_1080p + (pad_w * 2))
                    start_h = int(bh_1080p + (pad_h * 2))
                    
                    # Clamp to 1080p screen
                    start_x = max(0, start_x)
                    start_y = max(0, start_y)
                    if (start_x + start_w) > DISPLAY_WIDTH: start_w = DISPLAY_WIDTH - start_x
                    if (start_y + start_h) > DISPLAY_HEIGHT: start_h = DISPLAY_HEIGHT - start_y

                    start_box = (start_x, start_y, start_w, start_h)
                    tracker.init(frame_1080p, start_box)
                    
                    # Reset placement logic
                    candidate_shelf = None
                    candidate_lock_time = 0
                    last_center = None
            
            # 3. UPDATE TRACKER & PLACEMENT LOGIC (on 1080p frame)
            elif tracker:
                success, box = tracker.update(frame_1080p)
                
                if success:
                    (tx, ty, tw, th) = [int(v) for v in box]
                    cv2.rectangle(display_frame, (tx, ty), (tx + tw, ty + th), COLOR_RED, 2)
                    
                    center_x = tx + tw // 2
                    center_y = ty + th // 2
                    current_center = (center_x, center_y)
                    cv2.circle(display_frame, current_center, 5, COLOR_RED, -1)
                    
                    # --- CHECK SHELF LOCATION ---
                    hover_shelf = None
                    for i, sh_4k in enumerate(zm.zones.get("shelves", [])):
                        # Scale shelf to 1080p for collision check
                        (sx, sy, sw, sh_h) = scale_rect_to_display(sh_4k)
                        if (sx < center_x < sx + sw) and (sy < center_y < sy + sh_h):
                            hover_shelf = f"Shelf {i+1}"
                            break
                    
                    # --- CANDIDATE DETECTION (Dwell Logic) ---
                    # Calculate speed
                    speed = 0
                    if last_center:
                        dx = center_x - last_center[0]
                        dy = center_y - last_center[1]
                        speed = np.sqrt(dx*dx + dy*dy)
                    last_center = current_center

                    # Logic: If inside a shelf AND moving slowly (< 3 pixels/frame), it's a Candidate
                    if hover_shelf:
                        if speed < 5.0: # Threshold for "Stopped"
                            if candidate_shelf != hover_shelf:
                                # New candidate discovered
                                candidate_shelf = hover_shelf
                                candidate_lock_time = time.time()
                                print(f"👉 Candidate: {candidate_shelf}...")
                        
                        # Show status on screen
                        cv2.putText(display_frame, f"Hovering: {hover_shelf}", (tx, ty - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)
                    
                    cv2.putText(display_frame, f"Tracking: {tracking_item_name}", (tx, ty - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
                    
                else: 
                    # --- TRACKING LOST -> COMMIT ---
                    # The user removed their hand/object. Where was the last Candidate?
                    cv2.putText(display_frame, "Tracking Lost!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
                    
                    # Verify if we had a solid candidate
                    if candidate_shelf:
                         print(f"💾 LOGGED: {tracking_item_name} placed on {candidate_shelf} (Tracking Lost)")
                    else:
                         print(f"⚠️ LOST: {tracking_item_name} was not placed in any shelf.")
                    
                    # Reset everything
                    tracker = None
                    tracking_item_name = None
                    candidate_shelf = None
                    last_center = None

            # Check for Setup toggle
            if key == ord('s'):
                mode = "SETUP"
                setup_step = "BARCODE"
                current_rect = None
                print("Entering Setup Mode...")

        # Show the result
        cv2.imshow('Pantry Eye', display_frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

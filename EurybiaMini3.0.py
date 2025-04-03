#gradio?
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import diffusers
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import glob
import requests
from io import BytesIO
import time
import wikipedia
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from diffusers import DiffusionPipeline
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Eurybia Mini")
        self.root.geometry("800x600")

        # Set the theme
        style = ttk.Style()
        style.theme_use('clam')

        self.source_weights_path = "/Users/David/Downloads/WheelOfFortuneLab-DavidDriscoll/Eurybia1.3/mbari_315k_yolov8.pt"

        # Initialize variables
        self.bounding_boxes = []  # To store bounding boxes and class info
        self.original_image = None  # To store the original unedited image

        # Device detection
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.device = device

        # Initialize models
        self.initialize_models()

        # Show start page
        self.show_start_page()

    def initialize_models(self):
        # Initialize the RoBERTa model for question answering
        try:
            self.qa_pipeline = pipeline(
                "question-answering", model="deepset/roberta-base-squad2")
        except Exception as e:
            messagebox.showerror(
                "Error", f"Error initializing the RoBERTa model: {e}")
            print(f"Error initializing the RoBERTa model: {e}")
            self.qa_pipeline = None

        # Initialize the Gemma model
        try:
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-2-2b-it")
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b-it",
                device_map="auto",
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32
            )
        except Exception as e:
            messagebox.showerror(
                "Error", f"Error initializing the Gemma model: {e}")
            print(f"Error initializing the Gemma model: {e}")
            self.gemma_model = None

        # Initialize the depth estimation model using DiffusionPipeline exactly as per your example
        try:
            if self.device == 'cuda':
                self.depth_pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                    "prs-eth/marigold-depth-lcm-v1-0",
                    variant="fp16",
                    torch_dtype=torch.float16
                ).to('cpu')
            # For CPU or MPS devices
            else:
                self.depth_pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                    "prs-eth/marigold-depth-lcm-v1-0"
                ).to('cpu')
            self.depth_pipe.to(self.device)
        except Exception as e:
            messagebox.showerror(
                "Error", f"Error initializing the depth estimation model: {e}")
            print(f"Error initializing the depth estimation model: {e}")
            self.depth_pipe = None

        # Initialize the upscaling model
        try:
            # Adjusted the model to avoid the 404 error
            self.upscaler_model_path = 'weights/RealESRGAN_x4plus.pth'  # Ensure this path is correct
            if not os.path.exists(self.upscaler_model_path):
                # Download or prompt to download the model weights
                messagebox.showwarning(
                    "Model Not Found", f"Upscaling model weights not found at {self.upscaler_model_path}. Please download them.")
                self.upscaler = None
            else:
                # Define the model architecture
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=23, num_grow_ch=32, scale=4)

                # Initialize RealESRGANer
                self.upscaler = RealESRGANer(
                    scale=4,
                    model_path=self.upscaler_model_path,
                    model=model,
                    pre_pad=0,
                    half=(self.device == 'cuda'),
                    device=self.device
                )
        except Exception as e:
            messagebox.showerror(
                "Error", f"Error initializing the upscaling model: {e}")
            print(f"Error initializing the upscaling model: {e}")
            self.upscaler = None

    def show_start_page(self):
        # Create start page frame
        self.start_frame = ttk.Frame(self.root)
        self.start_frame.pack(fill=tk.BOTH, expand=True)

        # Welcome label
        welcome_label = ttk.Label(
            self.start_frame, text="Welcome to Eurybia", font=("Arial", 24, "bold"))
        welcome_label.pack(pady=20)

        # Load and display image
        try:
            # Replace with your actual image path
            image = Image.open(
                '/Users/David/Downloads/EurybiaRedSharkGeometricLogo-removebg.png')
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            image_label = ttk.Label(self.start_frame, image=photo)
            image_label.image = photo
            image_label.pack(pady=10)
        except Exception as e:
            print(f"Error loading image: {e}")
            # If image loading fails, display a placeholder label
            image_label = ttk.Label(
                self.start_frame, text="Image not found.", font=("Arial", 14))
            image_label.pack(pady=10)

        # Start button
        start_button = ttk.Button(
            self.start_frame, text="Start", command=self.start_app)
        start_button.pack(pady=20)

    def start_app(self):
        # Destroy start page and create main UI
        self.start_frame.destroy()
        self.create_ui()

    def create_ui(self):
        # Create main container
        main_frame = ttk.Frame(self.root, padding=(10, 10, 10, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Process Image button
        self.process_button = ttk.Button(
            buttons_frame, text="Process Image", command=self.start_processing)
        self.process_button.pack(side=tk.LEFT, padx=5)

        # Clear button
        self.clear_button = ttk.Button(
            buttons_frame, text="Clear", command=self.clear_display)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Info button
        self.info_button = ttk.Button(
            buttons_frame, text="Info", command=self.open_info_window)
        self.info_button.pack(side=tk.LEFT, padx=5)

        # Depth button
        self.depth_button = ttk.Button(
            buttons_frame, text="Depth", command=self.run_depth_prediction)
        self.depth_button.pack(side=tk.LEFT, padx=5)

        # Create image display frame
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.processed_image_label = ttk.Label(image_frame)
        self.processed_image_label.pack()

        # Detection info label
        self.detection_info_label = ttk.Label(
            main_frame, font=("Arial", 12))
        self.detection_info_label.pack(pady=5)

        # Store frames for later use
        self.image_frame = image_frame
        self.main_frame = main_frame

    def find_latest_screenshot(self):
        screenshots_dir = os.path.expanduser("~/Desktop/Screenshots")
        screenshot_extensions = ["png", "jpg", "jpeg"]
        screenshot_files = []

        for ext in screenshot_extensions:
            screenshot_files.extend(
                glob.glob(os.path.join(screenshots_dir, f"*.{ext}")))

        if not screenshot_files:
            raise FileNotFoundError(
                "No screenshots found in the 'Screenshots' folder on the Desktop.")

        latest_screenshot = max(screenshot_files, key=os.path.getctime)
        return latest_screenshot

    def process_screenshot(self):
        try:
            # Find the latest screenshot
            screenshot_path = self.find_latest_screenshot()

            # Load the screenshot using OpenCV
            self.original_image = cv2.imread(screenshot_path)

            # Load the YOLO model
            model = YOLO(self.source_weights_path)

            # Process the image using YOLO with a lower confidence threshold
            results = model.predict(
                source=self.original_image, conf=0.075)[0]  # Lowered the threshold

            self.bounding_boxes.clear()  # Clear previous bounding boxes

            # Draw bounding boxes on the image
            image = self.original_image.copy()
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_name = model.names[int(box.cls)]
                    confidence = box.conf.item() * 100  # Convert to percentage

                    # Save bounding box and class info
                    self.bounding_boxes.append({
                        "coords": (x1, y1, x2, y2),
                        "class_name": class_name,
                        "confidence": confidence
                    })

                    cv2.rectangle(image, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)
                    cv2.putText(image, f'{class_name} {confidence:.2f}%',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)

            # Save the processed image
            processed_image_path = "processed_screenshot.png"
            cv2.imwrite(processed_image_path, image)

            # Resize the image for display
            processed_image = Image.open(processed_image_path)
            processed_image.thumbnail((600, 400))
            processed_image = ImageTk.PhotoImage(processed_image)

            self.processed_image_label.configure(image=processed_image)
            self.processed_image_label.image = processed_image

            # Bind mouse click event to the image
            self.processed_image_label.bind("<Button-1>", self.on_image_click)

            # Display detection information
            if results.boxes is not None and len(results.boxes) > 0:
                detection_info = "\n".join(
                    [f'{model.names[int(box.cls)]}: {box.conf.item() * 100:.2f}%'
                     for box in results.boxes])
                return detection_info
            else:
                return "No detections found."

        except Exception as e:
            messagebox.showerror(
                "Error", f"Error processing screenshot: {e}")
            print(f"Error processing screenshot: {e}")
            return "No detections found."

    def start_processing(self):
        detection_info = "No detections found."
        start_time = time.time()
        while detection_info == "No detections found." and (time.time() - start_time) < 20:
            detection_info = self.process_screenshot()
            if detection_info == "No detections found.":
                print("No detections found, retrying...")

        self.detection_info_label.configure(text=detection_info)

    def clear_display(self):
        self.processed_image_label.configure(image='')
        self.processed_image_label.image = None
        self.detection_info_label.configure(text='')
        self.bounding_boxes.clear()
        self.original_image = None

    def open_info_window(self):
        info_window = tk.Toplevel(self.root)
        info_window.title("Info")
        info_window.geometry("500x500")  # Adjust size as needed

        # Make the info window scrollable
        info_canvas = tk.Canvas(info_window)
        info_scrollbar = ttk.Scrollbar(
            info_window, orient="vertical", command=info_canvas.yview)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_canvas.configure(yscrollcommand=info_scrollbar.set)

        info_frame = ttk.Frame(info_canvas)
        info_canvas.create_window((0, 0), window=info_frame, anchor="nw")

        info_frame.bind(
            "<Configure>",
            lambda e: info_canvas.configure(
                scrollregion=info_canvas.bbox("all")
            )
        )

        if self.detection_info_label and self.detection_info_label.cget("text") != "No detections found.":
            detected_classes = self.detection_info_label.cget(
                "text").split("\n")
            for detected_class in detected_classes:
                class_name = detected_class.split(":")[0]
                info_sub_frame = ttk.Frame(info_frame, padding=(10, 10))
                info_sub_frame.pack(pady=10, fill=tk.X)

                ttk.Label(info_sub_frame, text=class_name,
                          font=("Arial", 14, "bold")).pack()

                # Search for the class information
                try:
                    description = self.search_class_description(class_name)
                    img_url = self.search_class_image(class_name)
                    if img_url:
                        headers = {
                            'User-Agent': 'MyApp/1.0 (https://example.com/contact; myemail@example.com)'
                        }
                        response = requests.get(img_url, headers=headers)
                        img_data = response.content
                        img = Image.open(BytesIO(img_data))
                        img.thumbnail((200, 200))
                        img = ImageTk.PhotoImage(img)
                        img_label = ttk.Label(info_sub_frame, image=img)
                        img_label.image = img
                        img_label.pack(pady=5)

                    if description:
                        desc_label = ttk.Label(
                            info_sub_frame, text=description, wraplength=450, justify="left")
                        desc_label.pack()
                except Exception as e:
                    ttk.Label(info_sub_frame,
                              text=f"Error fetching info for {class_name}: {e}").pack()
        else:
            ttk.Label(info_frame, text="No detections to show info for.").pack()

    def search_class_description(self, class_name):
        wikipedia.set_lang("en")
        wikipedia.set_rate_limiting(True)
        description = ""

        try:
            page = wikipedia.page(class_name)
            if page:
                description = page.content[:5000]  # Get more content
        except Exception as e:
            print(f"Error fetching description: {e}")

        return description

    def search_class_image(self, class_name):
        wikipedia.set_lang("en")
        wikipedia.set_rate_limiting(True)
        img_url = ""

        try:
            page = wikipedia.page(class_name)
            if page:
                for img in page.images:
                    if img.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        img_url = img
                        break
        except Exception as e:
            print(f"Error fetching image: {e}")

        return img_url

    def on_image_click(self, event):
        if self.original_image is None:
            return

        # Calculate click position
        click_x, click_y = event.x, event.y

        # Calculate scaling factors
        displayed_image_width = self.processed_image_label.winfo_width()
        displayed_image_height = self.processed_image_label.winfo_height()
        original_image_width = self.original_image.shape[1]
        original_image_height = self.original_image.shape[0]
        scale_x = original_image_width / displayed_image_width
        scale_y = original_image_height / displayed_image_height

        # Scale the click position to the original image size
        original_click_x = int(click_x * scale_x)
        original_click_y = int(click_y * scale_y)

        for box in self.bounding_boxes:
            x1, y1, x2, y2 = box["coords"]

            # Check if the scaled click is within the bounding box
            if x1 <= original_click_x <= x2 and y1 <= original_click_y <= y2:
                self.open_inspect_window(box)
                break

    def open_inspect_window(self, box_info):
        inspect_window = tk.Toplevel(self.root)
        inspect_window.title("Inspect")
        inspect_window.geometry("600x800")

        class_name = box_info["class_name"]
        confidence = box_info["confidence"]
        x1, y1, x2, y2 = box_info["coords"]

        # Create main frame
        main_frame = ttk.Frame(inspect_window, padding=(10, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Fetch the description (context)
        try:
            description = self.search_class_description(class_name)
        except Exception as e:
            description = ""
            print(f"Error fetching description: {e}")

        # Ask Eurybia button
        ask_button = ttk.Button(
            main_frame, text="Ask Eurybia", command=lambda: self.open_ask_eurybia_window(description))
        ask_button.pack(pady=10)

        # Enhance Image button
        cropped_image = self.original_image[y1:y2, x1:x2]
        cropped_image_pil = Image.fromarray(
            cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        enhance_button = ttk.Button(
            main_frame, text="Enhance Image", command=lambda: self.enhance_image(cropped_image_pil, description))
        enhance_button.pack(pady=10)

        # Display the cropped image
        cropped_image_pil.thumbnail((300, 300))
        cropped_photo = ImageTk.PhotoImage(cropped_image_pil)

        img_label = ttk.Label(main_frame, image=cropped_photo)
        img_label.image = cropped_photo
        img_label.pack(pady=10)

        # Display class info with confidence as percentage
        info_label = ttk.Label(
            main_frame, text=f"Class: {class_name}\nConfidence: {confidence:.2f}%", font=("Arial", 12))
        info_label.pack(pady=10)

        # Description area with scrollbar
        if description.strip():
            desc_frame = ttk.Frame(main_frame)
            desc_frame.pack(fill=tk.BOTH, expand=True, pady=5)

            desc_scrollbar = ttk.Scrollbar(desc_frame)
            desc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            desc_text = tk.Text(desc_frame, wrap="word",
                                yscrollcommand=desc_scrollbar.set, height=5)
            desc_text.insert(tk.END, description)
            desc_text.config(state=tk.DISABLED)  # Make it read-only
            desc_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            desc_scrollbar.config(command=desc_text.yview)
        else:
            ttk.Label(main_frame, text="No description available.").pack()

    def open_ask_eurybia_window(self, description):
        ask_window = tk.Toplevel(self.root)
        ask_window.title("Ask Eurybia")
        ask_window.geometry("600x400")

        # Create main frame
        main_frame = ttk.Frame(ask_window, padding=(10, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Question label
        question_label = ttk.Label(main_frame, text="Ask Eurybia:", font=("Arial", 12, "bold"))
        question_label.pack(pady=5)

        question_entry = ttk.Entry(main_frame, width=50)
        question_entry.pack(pady=5)

        def get_answer():
            question = question_entry.get()
            if question.strip() == "":
                messagebox.showwarning("Input Required", "Please enter a question.")
                return
            context = description
            if not context or context.strip() == "":
                messagebox.showwarning("No Context", "Sorry, Eurybia does not have enough information about this creature")
                return
            try:
                answer = self.qa_pipeline(question=question, context=context)

                # Check if the answer is empty or whitespace
                if not answer['answer'].strip():
                    display_answer = "Unknown"
                else:
                    display_answer = answer['answer']

                # Clear previous answer
                answer_text.config(state=tk.NORMAL)
                answer_text.delete(1.0, tk.END)
                # Display the answer
                answer_text.insert(tk.END, f"Answer: {display_answer}")
                answer_text.config(state=tk.DISABLED)
            except Exception as e:
                answer_text.config(state=tk.NORMAL)
                answer_text.delete(1.0, tk.END)
                answer_text.insert(tk.END, f"Error: {e}")
                answer_text.config(state=tk.DISABLED)

        submit_button = ttk.Button(main_frame, text="Submit", command=get_answer)
        submit_button.pack(pady=5)

        # Answer area with scrollbar
        answer_frame = ttk.Frame(main_frame)
        answer_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        answer_scrollbar = ttk.Scrollbar(answer_frame)
        answer_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        answer_text = tk.Text(answer_frame, wrap="word", yscrollcommand=answer_scrollbar.set, height=10)
        answer_text.config(state=tk.DISABLED)
        answer_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        answer_scrollbar.config(command=answer_text.yview)

    def enhance_image(self, image_pil, description):
        try:
            if self.upscaler is None:
                messagebox.showwarning(
                    "Upscaler Not Available", "Upscaling model is not initialized.")
                return

            enhance_window = tk.Toplevel(self.root)
            enhance_window.title("Enhanced Image")

            # Display area for enhanced image
            img_label = ttk.Label(enhance_window)
            img_label.pack(fill=tk.BOTH, expand=True)

            # We need to keep references to the images
            self.enhanced_image = None
            self.resized_photo = None

            def run_enhancement():
                try:
                    input_image = image_pil.convert("RGB")
                    img = np.array(input_image)

                    # Run the model to enhance the image
                    output, _ = self.upscaler.enhance(img, outscale=4)

                    enhanced_image = Image.fromarray(output)

                    # Save the enhanced image
                    self.enhanced_image = enhanced_image

                    # Schedule the GUI update in the main thread
                    self.root.after(0, update_gui)

                except Exception as e:
                    # Schedule the error message display in the main thread
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Error during image enhancement: {e}"))

            def update_gui():
                # Set window size to match image size
                width, height = self.enhanced_image.size
                enhance_window.geometry(f"{width}x{height}")

                # Resize image to fit current window size
                resized_image = self.enhanced_image.copy()
                self.resized_photo = ImageTk.PhotoImage(resized_image)
                img_label.config(image=self.resized_photo)

                # Bind the resize event
                def on_resize(event):
                    # Get the size of the img_label
                    new_width = img_label.winfo_width()
                    new_height = img_label.winfo_height()

                    if new_width <= 0 or new_height <= 0:
                        return

                    # Calculate aspect ratio
                    original_width, original_height = self.enhanced_image.size
                    aspect_ratio = original_width / original_height

                    if new_width / new_height > aspect_ratio:
                        # Window is wider than image aspect ratio
                        new_width = int(new_height * aspect_ratio)
                    else:
                        # Window is taller than image aspect ratio
                        new_height = int(new_width / aspect_ratio)

                    # Resize the image to fit the new label size
                    resized_image = self.enhanced_image.resize(
                        (new_width, new_height), Image.LANCZOS)
                    self.resized_photo = ImageTk.PhotoImage(resized_image)
                    img_label.config(image=self.resized_photo)

                # Bind the configure event
                img_label.bind("<Configure>", on_resize)

            # Start the enhancement in a new thread
            threading.Thread(target=run_enhancement).start()

        except Exception as e:
            messagebox.showerror(
                "Error", f"Could not initialize the upscaling model: {e}")

    def run_depth_prediction(self):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please process an image first.")
            return

        # Convert the original image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))

        try:
            # Prepare the image
            input_image = image_pil.convert("RGB")

            # Run the depth pipeline
            result = self.depth_pipe(input_image)

            # Access the depth prediction
            depth_prediction = result.prediction

            # Visualize the depth map
            vis_depth = self.depth_pipe.image_processor.visualize_depth(depth_prediction)
            vis_depth_image = vis_depth[0]  # Assuming the first image is the one you want

            # Display the depth result
            self.show_depth_result(vis_depth_image)

        except Exception as e:
            messagebox.showerror("Error", f"Error during depth prediction: {e}")
            print(f"Error during depth prediction: {e}")


    def show_depth_result(self, depth_image):
        depth_window = tk.Toplevel(self.root)
        depth_window.title("Depth Prediction")

        # Display the depth image
        depth_photo = ImageTk.PhotoImage(depth_image)
        depth_label = ttk.Label(depth_window, image=depth_photo)
        depth_label.image = depth_photo
        depth_label.pack()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")

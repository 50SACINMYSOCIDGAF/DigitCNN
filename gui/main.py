import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageGrab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import subprocess
import json
from PIL import ImageTk  # Added this for Tkinter image handling
import io
from PIL import Image, ImageDraw, ImageGrab, ImageTk


class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")

        # Create an in-memory image for drawing
        self.image = Image.new('RGB', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack(pady=20)

        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_coordinates)

        # Initialize drawing variables
        self.old_x = None
        self.old_y = None

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text='Clear', command=self.clear_canvas).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text='Recognize', command=self.recognize_digit).pack(side=tk.LEFT, padx=10)

        # Animation window
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def paint(self, event):
        if self.old_x and self.old_y:
            # Draw on both canvas and image
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=20, fill='white', capstyle=tk.ROUND,
                                    smooth=tk.TRUE)
            self.draw.line([self.old_x, self.old_y, event.x, event.y],
                           fill='white', width=20)
        self.old_x = event.x
        self.old_y = event.y

    def reset_coordinates(self, event):
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('RGB', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def get_digit_image(self):
        # Convert to grayscale and resize
        image = self.image.convert('L')
        image = image.resize((28, 28))

        # Convert to numpy array and normalize
        digit = np.array(image)

        # Center the data to match MNIST distribution
        # MNIST data is centered around 0.1307 with std of 0.3081
        digit = (digit / 255.0 - 0.1307) / 0.3081

        # Debug: Print some statistics about the image
        print(f"Image statistics: min={digit.min()}, max={digit.max()}, mean={digit.mean()}")

        # Check if the image is mostly empty
        if np.abs(digit).mean() < 0.1:  # Adjusted threshold for centered data
            print("Warning: Empty drawing detected. Please draw something first.")
            return None

        return digit.flatten()

    def recognize_digit(self):
        # Get the drawn digit
        digit = self.get_digit_image()
        if digit is None:
            return

        try:
            # Print the data we're sending
            input_data = json.dumps(digit.tolist()).encode()
            print("Sending to Haskell:", input_data[:100], "...")  # Print first 100 chars

            process = subprocess.Popen([
                                           r'C:\Users\noah\digit-cnn\dist-newstyle\build\x86_64-windows\ghc-9.4.8\digit-cnn-0.1.0.0\x\digit-cnn\build\digit-cnn\digit-cnn.exe'],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       cwd=r'C:\Users\noah\digit-cnn')  # Add this line to set working directory

            # Send data and get response
            output_data, error_data = process.communicate(input_data)

            # Print raw output and error
            print("Raw output:", output_data)
            print("Error output:", error_data)

            if error_data:
                print("Error from Haskell program:", error_data.decode())
                return

            if not output_data:
                print("No output received from Haskell program")
                return

            # Parse the response
            result = json.loads(output_data.decode())
            print("Parsed result:", result)

            # Animate the recognition process
            self.animate_recognition([
                result['layer_output'],  # First layer output
                result['predictions']  # Final predictions
            ])

        except Exception as e:
            print(f"Error during recognition: {str(e)}")
            import traceback
            traceback.print_exc()

    def animate_recognition(self, cnn_data):
        # Setup the animation plot
        self.ax.clear()

        def update(frame):
            self.ax.clear()
            layer_data = cnn_data[frame]

            if frame == 0:  # Convolutional layer
                # Reshape the layer output into a square for visualization
                size = int(np.sqrt(len(layer_data)))
                layer_data = np.array(layer_data).reshape(size, size)
                self.ax.imshow(layer_data, cmap='viridis')
                self.ax.set_title('Layer 1 Activation')
            else:  # Final prediction
                self.ax.bar(range(10), layer_data)
                self.ax.set_title('Digit Probabilities')
                self.ax.set_xticks(range(10))

        anim = FuncAnimation(self.fig, update, frames=len(cnn_data),
                             interval=500, repeat=False)
        plt.show()


if __name__ == '__main__':
    root = tk.Tk()
    app = DigitDrawer(root)
    root.mainloop()
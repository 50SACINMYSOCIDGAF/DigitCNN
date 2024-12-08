import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import subprocess
import json


class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")

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
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=20, fill='white', capstyle=tk.ROUND,
                                    smooth=tk.TRUE)
        self.old_x = event.x
        self.old_y = event.y

    def reset_coordinates(self, event):
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        self.canvas.delete('all')

    def get_digit_image(self):
        # Convert canvas to image
        x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
        y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        image = ImageGrab.grab().crop((x, y, x1, y1))
        image = image.resize((28, 28)).convert('L')

        # Normalize pixel values
        digit = np.array(image) / 255.0
        return digit

    def recognize_digit(self):
        # Get the drawn digit
        digit = self.get_digit_image()

        # Call Haskell CNN through FFI
        result = self.process_with_cnn(digit)

        # Animate the recognition process
        self.animate_recognition(result)

    def animate_recognition(self, cnn_data):
        # Setup the animation plot
        self.ax.clear()

        def update(frame):
            self.ax.clear()
            layer_data = cnn_data[frame]

            if frame < len(cnn_data) - 1:  # Convolutional layers
                self.ax.imshow(layer_data, cmap='viridis')
                self.ax.set_title(f'Layer {frame + 1} Activation')
            else:  # Final prediction
                self.ax.bar(range(10), layer_data)
                self.ax.set_title('Digit Probabilities')
                self.ax.set_xticks(range(10))

        anim = FuncAnimation(self.fig, update, frames=len(cnn_data),
                             interval=500, repeat=False)
        plt.show()

    def process_with_cnn(self, digit):
        # Convert numpy array to list for JSON serialization
        digit_list = digit.tolist()

        # Call Haskell program with digit data
        process = subprocess.Popen(['runhaskell', 'DigitCNN.hs'],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)

        # Send digit data to Haskell
        input_data = json.dumps(digit_list).encode()
        output_data = process.communicate(input_data)[0]

        # Parse the response
        return json.loads(output_data.decode())


if __name__ == '__main__':
    root = tk.Tk()
    app = DigitDrawer(root)
    root.mainloop()
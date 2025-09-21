import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
from model import NeuralNetwork
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNetwork(input_shape=784, output_shape=10).to(device)
model.load_state_dict(torch.load("mnist_model_best.pth", map_location=device))
model.eval()

class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Recognizer")

        self.canvas = tk.Canvas(master, width=280, height=280, bg="white")
        self.canvas.pack()

        self.button_predict = tk.Button(master, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.button_clear = tk.Button(master, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.label_pred = tk.Label(master, text="Draw a digit and click Predict", font=("Helvetica", 16))
        self.label_pred.pack()

        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x-8), (event.y-8)
        x2, y2 = (event.x+8), (event.y+8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
    
    def predict(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).astype(np.float32)/255.0
        img_tensor = torch.tensor(img_array).view(1, 28*28).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()

        self.label_pred.config(text=f"Predicted digit: {pred}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()




import tkinter as tk
from PIL import Image, ImageTk
import glob

image_paths = glob.glob("./img_align_celeba/*.jpg")

# helpers class
class ImageLabeler:
    def __init__(self, image_paths, frame):
        self.image_paths = image_paths
        self.index = 0
        self.Image = None
        self.frame = frame
        # opening and loading the very first image
        self.Image = Image.open(image_paths[self.index])
        self.convertedImage = ImageTk.PhotoImage(self.Image)

        self.imageLabel = tk.Label(self.frame, image=self.convertedImage)

        # Add counter above the image
        self.counterLabel = tk.Label(self.frame, text=f"1 / {len(self.image_paths)}")
        self.counterLabel.pack(anchor="e") # "e" = east = right side
        
    
    def next_image(self):
        self.index += 1
        self.Image = Image.open(self.image_paths[self.index])
        self.convertedImage = ImageTk.PhotoImage(self.Image)
        self.imageLabel.config(image=self.convertedImage)
        self.counterLabel.config(text=f"{self.index + 1} / {len(self.image_paths)}")
    
    def pack(self):
        self.imageLabel.pack()



# Create the main window
root = tk.Tk()
root.title("Image Labeler")


# Set window size
width = 1200
height = 900

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate center position
x = (screen_width - width) // 2
y = (screen_height - height) // 2

# Set geometry: "widthheight+x+y"
root.geometry(f"{width}x{height}+{x}+{y}")



# Create the Frame to group and vertically center the image and the button elements
frame = tk.Frame(root)
frame.pack(expand=True)


# Create instance of image class and pass it the image paths
labeler = ImageLabeler(image_paths=image_paths, frame=frame)
labeler.pack()

# Put the button inside the frame (not root)
button = tk.Button(frame, text="Next", command=labeler.next_image)
button.pack()

# Start the event loop (keeps window open)
root.mainloop()



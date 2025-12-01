
import tkinter as tk
from PIL import Image, ImageTk
import glob
import csv
import os

image_paths = sorted(glob.glob("./img_align_celeba/*.jpg"))

# helpers class
class ImageLabeler:
    def __init__(self, image_paths, frame):
        self.image_paths = image_paths
        self.index = 0
        self.Image = None
        self.frame = frame

        # Check if CSV already exists, to continue
        if os.path.exists("ratings.csv"):
            with open("ratings.csv", "r") as f:
                self.index = sum(1 for line in f)

        # opening and loading the very first image
        self.Image = Image.open(image_paths[self.index])
        self.convertedImage = ImageTk.PhotoImage(self.Image)

        self.imageLabel = tk.Label(self.frame, image=self.convertedImage)

        # Add counter above the image
        self.counterLabel = tk.Label(self.frame, text=f"{self.index + 1} / {len(self.image_paths)}")
        self.counterLabel.pack(anchor="e") # "e" = east = right side

        # Add file name
        self.filenameLabel = tk.Label(self.frame, text=f"{self.image_paths[self.index]}")
        self.filenameLabel.pack(anchor="e")

        # Rating buttons
        self.buttonFrame = tk.Frame(self.frame)
        
        for i in range(1, 11):
            btn = tk.Button(self.buttonFrame, text=str(i), command=lambda x=i: self.rate(x))
            btn.pack(side="left")
        
        
    
    def next_image(self):
        self.index += 1
        if self.index >= len(self.image_paths):
            print("Done Labeling!")
            root.quit() # or root.destroy()
            return

        self.Image = Image.open(self.image_paths[self.index])
        self.convertedImage = ImageTk.PhotoImage(self.Image)
        self.imageLabel.config(image=self.convertedImage)
        self.counterLabel.config(text=f"{self.index + 1} / {len(self.image_paths)}")

        self.filenameLabel.config(text=f"{self.image_paths[self.index]}")
    
    def pack(self):
        self.imageLabel.pack()
        self.buttonFrame.pack()

    def rate(self, score):
        # save the rating
        with open("ratings.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.image_paths[self.index], score])

        print(f"Rated {self.image_paths[self.index]}: {score}")
        self.next_image()



# Create the main window
root = tk.Tk()
root.title("Image Labeler")


# Set window size
width = 900
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


# Start the event loop (keeps window open)
root.mainloop()



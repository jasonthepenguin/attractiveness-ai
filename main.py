
import tkinter as tk
from PIL import Image, ImageTk


# helper functions

def next_image():
    # get the next image
    print("Hello")

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

# Opening and converting image file to be ready for use in Tkinter
firstImage = Image.open("./img_align_celeba/000954.jpg")
convertedImage = ImageTk.PhotoImage(firstImage)


# Create the Frame to group and vertically center the image and the button elements
frame = tk.Frame(root)
frame.pack(expand=True)

# Creating the label and image to display inside the frame

# Put the label inside the frame (not root)
label = tk.Label(frame, image=convertedImage)
label.pack()

# Put the button inside the frame (not root)
button = tk.Button(frame, text="Next", command=next_image)
button.pack()

# Start the event loop (keeps window open)
root.mainloop()



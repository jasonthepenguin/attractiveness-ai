
import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Image Labler")


# Set window size
width = 800
height = 600

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate center position
x = (screen_width - width) // 2
y = (screen_height - height) // 2

# Set geometry: "widthheight+x+y"
root.geometry(f"{width}x{height}+{x}+{y}")

# Start the event loop (keeps window open)
root.mainloop()
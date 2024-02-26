import tkinter as tk

def say_hello():
    label.config(text="Hello, World!")

root = tk.Tk()
root.title("Hello World App")

label = tk.Label(root, text="Click the button to say hello!")
label.pack(pady=10)

button = tk.Button(root, text="Say Hello", command=say_hello)
button.pack(pady=5)

root.mainloop()
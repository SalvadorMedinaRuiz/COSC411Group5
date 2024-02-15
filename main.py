import os
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL.ImageTk import PhotoImage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

register_matplotlib_converters()


class FruitPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Machine Learning App")
        self.df = None  # Initialize DataFrame
        self.fruits = ""  # Initialize available fruits string

        # Get the path to the image file
        image_path = os.path.abspath("fruit.jpg")

        # Load the image as an instance variable
        self.bg = PhotoImage(file=image_path)
        label_background = tk.Label(root, image=self.bg)
        label_background.place(x=0, y=0, relwidth=1, relheight=1)

        # Welcome Message
        welcome_label = tk.Label(root, text="Welcome to the fruit price prediction App", font=("Helvetica", 16))
        welcome_label.pack(pady=20)

        # Main Menu
        main_menu = tk.Menu(root)
        root.config(menu=main_menu)

        # File Menu
        file_menu = tk.Menu(main_menu, tearoff=0)
        main_menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)

        # Fruit Recognition Button
        display_data_button = tk.Button(root, text="Display data", command=self.display_data,
                                        font=("Helvetica", 12))
        display_data_button.pack(pady=20)

        # Fruit Recognition Button
        display_general_stats_button = tk.Button(root, text="Display General Stats", command=self.stats_for_all_items,
                                               font=("Helvetica", 12))
        display_general_stats_button.pack(pady=20)

        # Price Prediction Button
        price_prediction_button = tk.Button(root, text="Price Prediction", command=self.price_prediction,
                                            font=("Helvetica", 12))
        price_prediction_button.pack(pady=20)

    def load_data(self):
        # Prompt user to select a file
        file_path = filedialog.askopenfilename(title="Select a CSV file",
                                               filetypes=[("CSV files", "*.csv")])

        # Check if the user selected a file
        if file_path:
            # Load data from the selected file
            self.df = pd.read_csv(file_path)
            self.fruits = ""
            available_fruits = self.df['Fruit Type'].unique()
            for fruit in available_fruits:
                self.fruits += fruit + " "
            messagebox.showinfo("Data Loaded", "Data loaded successfully from:\n{}".format(file_path))
        else:
            messagebox.showinfo("No File Selected", "No file selected. Please choose a CSV file.")

    def display_data(self):
        if self.df is None:
            tk.messagebox.showinfo("Data Not Loaded", "Please load data before using this fuctunality.")
            return

    def stats_for_all_items(self):
        if self.df is None:
            tk.messagebox.showinfo("Data Not Loaded", "Please load data before viewing general statistics.")
            return

    def price_prediction(self):
        # Check if data is loaded
        if self.df is None:
            messagebox.showinfo("Data Not Loaded", "Please load data before performing price prediction.")
            return

        # Display available fruit options
        selected_fruit, selected_algorithm = self.ask_fruit()

        # Check if the entered fruit is valid
        if selected_fruit:
            # Perform price prediction for the selected fruit based on the selected algorithm
            if selected_algorithm == "Linear Regression":
                self.linear_regression_prediction(selected_fruit)
        else:
            messagebox.showinfo("Invalid Fruit",
                                "Invalid fruit selection. Please choose from the available fruits.")

    def ask_fruit(self):
        # Create a Toplevel window for user input
        top = tk.Toplevel(self.root)
        top.title("Fruit Selection")

        label = tk.Label(top, text="Our system found the following fruits in your file:", font="Helvetica")
        label.pack(pady=2)

        label = tk.Label(top, text=self.fruits, font="Helvetica", wraplength=200)
        label.pack(pady=10)

        label = tk.Label(top, text="Enter one of the fruits for price prediction:", font="Helvetica")
        label.pack(pady=10)

        # Entry widget for user input
        fruit_var = tk.StringVar()
        entry = tk.Entry(top, textvariable=fruit_var)
        entry.pack(pady=10)

        label = tk.Label(top, text="Enter the algorithm you want to use for price prediction:", font="Helvetica")
        label.pack(pady=10)
        algorithm_var = tk.StringVar(top)
        algorithms = ["Linear Regression"]
        algorithm_var.set(algorithms[0])
        algorithm_menu = tk.OptionMenu(top, algorithm_var, *algorithms)
        algorithm_menu.pack(pady=2)

        # Button to submit the input
        submit_button = tk.Button(top, text="Submit", command=lambda: top.destroy())
        submit_button.pack(pady=10)

        # Wait for the Toplevel window to be closed
        top.wait_window()

        # Get the user input
        selected_fruit = fruit_var.get()
        selected_algorithm = algorithm_var.get()

        return selected_fruit, selected_algorithm

    def linear_regression_prediction(self, selected_fruit):
        # Filter the data for the selected fruit
        fruit_data = self.df[self.df['Fruit Type'] == selected_fruit]

        # Extract features (X) and target variable (y)
        X = fruit_data[['Date', 'Quantity Sold']]
        y = fruit_data['Price per Unit']

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(x_train[['Quantity Sold']], y_train)
        lr_predictions = lr_model.predict(x_test[['Quantity Sold']])

        fig, axes = plt.subplots()

        # Scatter plot for Linear Regression
        axes.scatter(x_test['Quantity Sold'], y_test, color='black', label='Actual Prices')
        axes.scatter(x_test['Quantity Sold'], lr_predictions, color='blue', label='Linear Regression Predictions')
        axes.set_title('Linear Regression')
        axes.set_xlabel('Quantity Sold')
        axes.set_ylabel('Price per Unit')
        axes.legend()

        # Create a new window for displaying the plots
        output_window = tk.Toplevel(self.root)
        output_window.title("Price Prediction Output")

        # Embed the Matplotlib plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=output_window)
        canvas.get_tk_widget().pack()

        # Display the window
        output_window.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = FruitPredictionApp(root)
    root.geometry("800x500")
    root.maxsize(1024, 1024)
    root.mainloop()
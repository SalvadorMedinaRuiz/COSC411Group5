import os
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL.ImageTk import PhotoImage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
register_matplotlib_converters()


class FruitPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Machine Learning App")
        self.df = None  # Initialize DataFrame
        self.fruits = ""  # Initialize available fruits string
        style = ttk.Style()
        style.theme_use('clam')  # Using 'clam' for its modern appearance

        # Get the path to the image file
        image_path = os.path.abspath("fruit.jpg")


        # Label style with light text that will stand out on the dark background
        style.configure('TLabel', foreground='white',font=("Helvetica", 25, "bold"), background="#36454F")

        # Button style with a contrasting color and hover effects
        style.configure('TButton', font=('Helvetica', 16, 'bold'), foreground='white', background="#283747",
                        borderwidth=1)
        style.map('TButton',
                  foreground=[('pressed', 'white'), ('active', 'white')],
                  background=[('pressed', 'black'), ('active', '#3E5F7B')],
                  relief=[('pressed', 'sunken'), ('!pressed', 'raised')])

        # Load the image as an instance variable
        self.bg = PhotoImage(file=image_path)
        label_background = tk.Label(root, image=self.bg)
        label_background.place(x=0, y=0, relwidth=1, relheight=1)

        # Welcome Message
        welcome_label = ttk.Label(root, text="Welcome to the fruit price prediction App", style="TLabel")
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
        display_data_button = ttk.Button(root, text="Display data", command=self.display_data,
                                         style="TButton")
        display_data_button.pack(pady=20)

        # Fruit Recognition Button
        display_general_stats_button = ttk.Button(root, text="Display General Stats", command=self.stats_for_all_items,
                                                style="TButton")
        display_general_stats_button.pack(pady=20)

        # Price Prediction Button
        price_prediction_button = ttk.Button(root, text="Price Prediction", command=self.price_prediction,
                                             style="TButton")
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

        # Create a top-level window to display the data
        data_window = tk.Toplevel(self.root)
        data_window.title("Display Data")
        data_window.configure(bg='#283747')

        # Create a treeview widget
        tree = ttk.Treeview(data_window)

        # Define the column headings
        tree['columns'] = list(self.df.columns)
        tree.column("#0", width=0, stretch=tk.NO)
        for col in self.df.columns:
            tree.column(col, anchor=tk.CENTER, width=80)
            tree.heading(col, text=col, anchor=tk.CENTER)

        # Insert the data into the treeview
        for index, row in self.df.iterrows():
            tree.insert("", index, text="", values=list(row))

        # Pack the treeview into the window
        tree.pack(side=tk.TOP, fill=tk.X)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(data_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def stats_for_all_items(self):
        if self.df is None:
            tk.messagebox.showinfo("Data Not Loaded", "Please load data before viewing general statistics.")
            return
        
        # Create a new window for displaying statistics
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Statistics for All Items")

        df1 = self.df.copy()

        # Calculate profit for each fruit
        df1['Profit'] = df1['Quantity Sold'] * df1['Price per Unit']

        # Create a single figure for all plots
        fig, (pie_ax, bar_ax, table_ax) = plt.subplots(1, 3, figsize=(20, 8))

        # Calculate total profit, and total quantity sold for each fruit
        fruit_profit = df1.groupby('Fruit Type')['Profit'].sum()
        total_quantity_sold_by_fruit = df1.groupby('Fruit Type')['Quantity Sold'].sum().reset_index()

        # Identify fruits with less than 3% profit
        low_profit_fruits = fruit_profit[fruit_profit / fruit_profit.sum() < 0.03].index

        # Replace those fruits with 'Others'
        df1.loc[df1['Fruit Type'].isin(low_profit_fruits), 'Fruit Type'] = 'Others'

        # Plot pie chart for distribution of fruit types based on profit
        df1.groupby('Fruit Type')['Profit'].sum().plot.pie(autopct='%1.1f%%', ax=pie_ax)
        pie_ax.set_title('Distribution of Fruit Types based on Sales')
        pie_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Plot bar chart for Quantity Sold by Fruit Type using the aggregated data
        sns.barplot(x='Fruit Type', y='Quantity Sold', data=total_quantity_sold_by_fruit, ax=bar_ax)
        bar_ax.set_title('Quantity Sold by Fruit Type')
        bar_ax.set_xticklabels(bar_ax.get_xticklabels(), rotation=45, ha='right')

        # Display profit statistics in a table
        fruit_profit_stats = df1.groupby('Fruit Type')['Profit'].sum().reset_index()
        fruit_profit_stats.columns = ['Fruit Type', 'Sales']
        table_ax.axis('off')  # Turn off axis for the table
        table_ax.table(cellText=fruit_profit_stats.values, colLabels=fruit_profit_stats.columns, cellLoc='center',
                       loc='center')

        # Display the canvas in stats_window
        canvas = FigureCanvasTkAgg(fig, master=stats_window)
        canvas.get_tk_widget().pack(side=tk.LEFT, padx=10)
        canvas.draw()

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
            elif selected_algorithm == "Logistic Regression":
                self.logistic_regression_prediction(selected_fruit)
            elif selected_algorithm == "Time Series":
                self.time_series(selected_fruit)
            elif selected_algorithm == "Random Forest":
                self.random_forest_regression(selected_fruit)
            elif selected_algorithm == "Decision Tree":
                self.decision_tree_regression(selected_fruit)
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
        algorithms = ["Linear Regression", "Logistic Regression", "Time Series", "Random Forest", "Decision Tree"]
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

    def logistic_regression_prediction(self, selected_fruit): #Salvador Medina-Ruiz
        #Filter the data for the selected fruit
        fruit_data = self.df[self.df['Fruit Type'] == selected_fruit]

        #Define a threshold for determining if the fruit sells more or not (this can change)
        sales_threshold = 100

        #Create a binary target variable based on the threshold
        fruit_data['Sells More'] = (fruit_data['Quantity Sold'] > sales_threshold).astype(int)

        #Extract features (X) and target variable (y) for the actual data stuff
        X = fruit_data[['Quantity Sold', 'Price per Unit', 'discount']]
        y = fruit_data['Sells More']

        #Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Logistic Regression stuff
        logisticRegr = LogisticRegression()
        logisticRegr.fit(x_train, y_train)
        logisticRegr_predictions = logisticRegr.predict(x_test)

        fig, axes = plt.subplots()

        #Scatter plot for Logistic Regression
        axes.scatter(x_test['Quantity Sold'], logisticRegr_predictions, color='blue', label='Logistic Regression Predictions')

        #Plot for the sigmoid curve thingy (idk how this works ask google)
        x_values = np.linspace(np.min(x_test['Quantity Sold']), np.max(x_test['Quantity Sold']), 100)
        z = logisticRegr.intercept_[0] + logisticRegr.coef_[0][0] * x_values  # Linear combination of features
        sigmoid_values = 1 / (1 + np.exp(-z))
        axes.plot(x_values, sigmoid_values, color='red', label='Sigmoid Curve')

        #Code to plot black dotted line at the mid point (0.5)
        axes.set_title('Probability of ' + selected_fruit + ' selling more next year with Logistic Regression')
        axes.set_xlabel('Quantity Sold')
        axes.set_ylabel('Sells More')
        axes.axhline(y=0.5, color='black', linestyle='--')

        axes.legend()

        #Create a new window for displaying the plots
        output_window = tk.Toplevel(self.root)
        output_window.title("Probability of fruit selling more next year")

        #Embed the Matplotlib plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=output_window)
        canvas.get_tk_widget().pack()

        #Display the window
        output_window.mainloop()
    def time_series(self, selected_fruit):#Uzair Mumtaz

        df = self.df[self.df['Fruit Type'] == selected_fruit]
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        df.index = df['Date']
        del df['Date']

        plt.ylabel("Price")
        sns.lineplot(df)
        plt.show()

    def random_forest_regression(self, selected_fruit): # Nathan
        # Filter the data for the selected fruit
        fruit_data = self.df[self.df['Fruit Type'] == selected_fruit]

        # Extract features (X) and target variable (y)
        X = fruit_data[['Date', 'Quantity Sold']]
        y = fruit_data['Price per Unit']

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest Regression
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(x_train[['Quantity Sold']], y_train)
        rf_predictions = rf_model.predict(x_test[['Quantity Sold']])

        fig, axes = plt.subplots()

        # Scatter plot for Random Forest Regression
        axes.scatter(x_test['Quantity Sold'], y_test, color='black', label='Actual Prices')
        axes.scatter(x_test['Quantity Sold'], rf_predictions, color='green', label='Random Forest Predictions')
        axes.set_title('Random Forest Regression')
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
        
    def decision_tree_regression(self, selected_fruit): #Nick Fiori

        # Filter the data for the selected fruit
        fruit_data = self.df[self.df['Fruit Type'] == selected_fruit]

        # Extract features (X) and target variable (y)
        X = fruit_data[['Quantity Sold']]
        y = fruit_data['Price per Unit']

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Decision Tree Regression
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_model.fit(x_train, y_train)
        dt_predictions = dt_model.predict(x_test)

        fig, axes = plt.subplots()

        # Scatter plot for Decision Tree Regression
        axes.scatter(x_test, y_test, color='black', label='Actual Prices')
        axes.scatter(x_test, dt_predictions, color='red', label='Decision Tree Predictions')
        axes.set_title('Decision Tree Regression')
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

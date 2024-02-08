import pandas as pd
import random
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)

# List of 20 popular fruits
fruits = ['Apple', 'Banana', 'Orange', 'Strawberry', 'Grapes', 'Watermelon', 'Mango', 'Pineapple', 'Peach', 'Cherry',
          'Kiwi', 'Pear', 'Blueberry', 'Raspberry', 'Blackberry', 'Cantaloupe', 'Plum', 'Apricot', 'Pomegranate', 'Lemon']

# Function to generate synthetic data for each day of the week for each fruit
def generate_fruit_sales_data(start_date, end_date):
    data = []

    current_date = start_date
    while current_date <= end_date:
        for fruit_type in fruits:
            if fruit_type == 'Apple':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(0.80, 1.40), 2)
            elif fruit_type == 'Banana':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(0.40, 0.80), 2)
            elif fruit_type == 'Orange':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(0.80, 1.50), 2)
            elif fruit_type == 'Strawberry':
                quantity_sold = random.randint(10, 40)
                price_per_unit = round(random.uniform(2.00, 5.00), 2)
            elif fruit_type == 'Grapes':
                quantity_sold = random.randint(20, 50)
                price_per_unit = round(random.uniform(2.50, 5.00), 2)
            elif fruit_type == 'Watermelon':
                quantity_sold = random.randint(20, 50)
                price_per_unit = round(random.uniform(5.00, 9.00), 2)
            elif fruit_type == 'Mango':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(1.50, 3.00), 2)
            elif fruit_type == 'Pineapple':
                quantity_sold = random.randint(30, 80)
                price_per_unit = round(random.uniform(3.50, 7.00), 2)
            elif fruit_type == 'Peach':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(1.50, 3.00), 2)
            elif fruit_type == 'Cherry':
                quantity_sold = random.randint(5, 20)
                price_per_unit = round(random.uniform(3.00, 6.00), 2)
            elif fruit_type == 'Kiwi':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(1.00, 2.00), 2)
            elif fruit_type == 'Pear':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(1.00, 2.50), 2)
            elif fruit_type == 'Blueberry':
                quantity_sold = random.randint(5, 20)
                price_per_unit = round(random.uniform(2.50, 5.00), 2)
            elif fruit_type == 'Raspberry':
                quantity_sold = random.randint(5, 20)
                price_per_unit = round(random.uniform(3.00, 6.00), 2)
            elif fruit_type == 'Blackberry':
                quantity_sold = random.randint(5, 20)
                price_per_unit = round(random.uniform(2.50, 5.00), 2)
            elif fruit_type == 'Cantaloupe':
                quantity_sold = random.randint(20, 50)
                price_per_unit = round(random.uniform(0.50, 1.00), 2)
            elif fruit_type == 'Plum':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(1.00, 2.50), 2)
            elif fruit_type == 'Apricot':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(2.00, 4.00), 2)
            elif fruit_type == 'Pomegranate':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(1.50, 3.50), 2)
            elif fruit_type == 'Lemon':
                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(0.30, 0.60), 2)
            else:

                quantity_sold = random.randint(80, 150)
                price_per_unit = round(random.uniform(0.70, 1.50), 2)

            
            data.append([current_date, fruit_type, quantity_sold, price_per_unit])

        current_date += timedelta(days=1)

    return data

# Set start and end dates
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)

# Generate data
sales_data = generate_fruit_sales_data(start_date, end_date)

# Create a DataFrame
columns = ['Date', 'Fruit Type', 'Quantity Sold', 'Price per Unit', 'discount']
df = pd.DataFrame(sales_data, columns=columns)

# Save to CSV
df.to_csv('fruit_sales_data.csv', index=False)
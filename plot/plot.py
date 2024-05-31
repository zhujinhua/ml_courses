"""
Author: jhzhu
Date: 2024/5/29
Description: 
"""

import matplotlib.pyplot as plt
import numpy as np

# Data preparation
sizes = [15, 30, 45, 10]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode the 1st slice

# Create a pie chart
plt.figure()
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Pie Chart Example')
plt.show()

# Data preparation
data = np.random.randn(1000)

# Create a histogram
plt.figure()
plt.hist(data, bins=30, color='green', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.show()

x = np.random.rand(100)
y = np.random.rand(100)

# Create a scatter plot
plt.figure()
plt.scatter(x, y, color='red', marker='o')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')
plt.grid(True)
plt.show()



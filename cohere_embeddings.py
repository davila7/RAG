import cohere
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)

# get the embeddings
words = ["Red", "Blood", "Sea"]
(p1, p2, p3) = co.embed(words).embeddings

x = [p1[0], p2[0], p3[0]] 
y = [p1[1], p2[1], p3[1]] 
z = [p1[2], p2[2], p3[2]]

# plot them
fig = plt.figure()

"""
This code creates a 3D scatter plot using Matplotlib.

The plot visualizes data points in a three-dimensional space,
with each point represented by its X, Y, and Z coordinates.
The axes are labeled accordingly for clarity.
"""

# Create a 3D subplot within the existing figure
ax = fig.add_subplot(111, projection='3d')

# Generate a scatter plot of the data points
ax.scatter(x, y, z)

# Set labels for each axis
ax.set_xlabel('X')  # Label for the X-axis
ax.set_ylabel('Y')  # Label for the Y-axis
ax.set_zlabel('Z')  # Label for the Z-axis

# 2d plot
plt.figure()
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')

plt.show()

# calculate the distances
distances = []
for i in range(len(words)):
    for j in range(i+1, len(words)):
        distance = np.linalg.norm(np.array(co.embed([words[i]]).embeddings[0]) - np.array(co.embed([words[j]]).embeddings[0]))
        distances.append((words[i], words[j], distance))



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

# 3d plot
ax = fig.add_subplot(projection='3d')

# plot points
ax.scatter(x, y, z)

# plot lines
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# show the plot
plt.show()
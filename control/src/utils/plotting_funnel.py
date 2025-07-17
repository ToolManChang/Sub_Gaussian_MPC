import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Define range for x, y, and z
z = np.linspace(-30, 30, 300)
y = np.linspace(-100, 100, 300)
dz = 200
dy = 200
dx = 200
Z, Y = np.meshgrid(z, y)

# Compute z values using the equation x^2 + y^2 = e^z
# Z = np.log(X**2 + Y**2 - 0.1) + 1
zeta = 0.0
value = 1 / 1600 - np.exp(-70 / 200)  # -1 + 1 / 1600 , -0.999375
shift = 0
X = dx * (np.log((Z / dz) ** 2 + (Y / dy) ** 2 - zeta - value) + shift)


# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, Z, cmap="viridis")

ax.invert_zaxis()
ax.contourf(Z, Y, -X, cmap=cm.coolwarm)


# Label axes
ax.set_xlabel("Z")
ax.set_ylabel("Y")
ax.set_zlabel("X")
# ax.set_zlim(-1, 1)

plt.title("Surface Plot of z^2 + y^2 = e^x")
plt.show()

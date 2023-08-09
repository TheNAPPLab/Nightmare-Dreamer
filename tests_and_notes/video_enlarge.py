from PIL import Image
import numpy as np


original_image = Image.open('CircleInteraction.png')

# Upscale the image to the desired size (e.g., 256x256)
desired_size = (256, 256)
upscaled_image = original_image.resize(desired_size, Image.BILINEAR)

# Convert the PIL image to a NumPy array
upscaled_array = np.array(upscaled_image)

# Save the upscaled image
upscaled_image.save('upscaled_image.png')

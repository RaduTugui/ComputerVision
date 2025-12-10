import numpy as np
import math 


def prepare_image(image: np.ndarray, width: int, height: int, x: int, y: int, size: int) -> \
        tuple[np.ndarray, np.ndarray]:
    if image.ndim < 3 or image.shape[-3] != 1:
        raise ValueError("image must have shape (1, H, W)")
    if width < 32 or height < 32 or size < 32:
        raise ValueError("width/height/size must be >= 32")
    if x < 0 or (x + size) > width:
        raise ValueError(f"x={x} and size={size} do not fit into the resized image width={width}")
    if y < 0 or (y + size) > height:
        raise ValueError(f"y={y} and size={size} do not fit into the resized image height={height}")
    

    # Copy image
    image = image.copy()

    # Check if we need to pad or crop the image
    if image.shape[1] > height:
        # Crop center area
        # if unequal crop one more at the start
        image = image[:, (image.shape[1] - height) // 2: (image.shape[1] - height) // 2 + height, :]
    else: 
        # Pad with same value as the image
        image = np.pad(image, ((0, 0), ((height - image.shape[1])//2, math.ceil((height - image.shape[1])/2)), (0, 0)), mode='edge')
    
    if image.shape[2] > width:
        # Crop center area
        image = image[:, :, (image.shape[2] - width) // 2: (image.shape[2] - width) // 2 + width]
    else:
        # Pad border in both directions with same value as the image
        image = np.pad(image, ((0, 0), (0, 0), ((width - image.shape[2])//2, math.ceil((width - image.shape[2])/2))), mode='edge')

    # Return subarea
    subarea = image[:, y:y + size, x:x + size]

    return image, subarea  


# Reproduce example
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    with Image.open('validation_images/cat.jpg') as im:  # This returns a PIL image
        #convert to grayscale
        im = im.convert('L')
        im = np.array(im) 
        #add channel dimension
        im = im[np.newaxis, ...]
    prepared, subarea = prepare_image(im, 1200, 1200, 400, 400, 150)
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(im[0], cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(prepared[0], cmap='gray')
    ax[1].set_title('Resized')
    ax[2].imshow(subarea[0], cmap='gray')
    ax[2].set_title('Subarea')
    fig.tight_layout()
    plt.show()
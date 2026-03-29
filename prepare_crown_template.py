import numpy as np
from PIL import Image


def make_white_transparent(input_path, output_path, threshold=220):
    """
    Removes near-white background from an image and makes it transparent.
    """
    # 1. Open the image and convert to RGBA (Red, Green, Blue, Alpha)
    try:
        img = Image.open(input_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Could not find the file '{input_path}'")
        return

    data = np.array(img)

    # 2. Define the condition for "near-white" pixels
    # A pixel is considered near-white if its R, G, and B values are all above the threshold
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    near_white_mask = (r > threshold) & (g > threshold) & (b > threshold)

    # 3. Set the alpha channel (transparency) to 0 for the near-white pixels
    data[:, :, 3][near_white_mask] = 0

    # 4. Convert the numpy array back to an image and save as PNG to preserve transparency
    result_img = Image.fromarray(data)
    result_img.save(output_path, format="PNG")
    print(f"Success! Saved transparent image to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Path to your input image
    INPUT_FILE = r"C:\Duas_mini_projekt\Obsolete\low_res_crown.jpg"
    # Target path for the output image (must be .png for transparency)
    OUTPUT_FILE = "low_res_crown.png"

    print("Processing image...")
    # You can adjust the threshold (0-255).
    # 255 = only pure white, 200 = includes off-white levels
    make_white_transparent(INPUT_FILE, OUTPUT_FILE, threshold=220)

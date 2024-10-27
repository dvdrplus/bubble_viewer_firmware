from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from skimage.color import label2rgb
from skimage.segmentation import watershed, clear_border
import numpy as np
from scipy import ndimage
import cv2
#from matplotlib import pyplot as plt
import time
import os
from scipy import ndimage
import pandas as pd
import json
from multiprocessing import Pool

"""
These functions are maintained in the TECHNOLOGY\PRODUCTS\data_platform\api\app\bubble_analyzer\helpers.py file.
This was pulled 2024-10-27
"""

def sauter_mean_diameter(bubble_diameters):
    sum_d3 = sum([d**3 for d in bubble_diameters])
    sum_d2 = sum([d**2 for d in bubble_diameters])
    return sum_d3 / sum_d2


def process_file(args):
    dir_path, file, settings = args
    # try:
    bubbles_array = []
    print(f"Processing file {file}...")

    image_path = os.path.join(dir_path, file)
    # Load the image with OpenCV
    image = cv2.imread(image_path)

    # Check if 'segmented' directory exists, if not, create it
    segmented_dir = os.path.join(dir_path, "segmented")
    if not os.path.exists(segmented_dir):
        os.makedirs(segmented_dir, exist_ok=True)

    # Split the string at the last occurrence of '.jpg'
    new_filename = os.path.basename(image_path)

    # Add '_segmented.jpg' to the filename and adjust the path to include the 'segmented' directory
    new_image_path = os.path.join(segmented_dir, new_filename)

    # Call analyze function on the image
    bubbles = analyze_image(
        image, settings, new_image_path, output_image=settings["OUTPUT_IMAGE"]
    )

    print(f"n_bubbles found: {len(bubbles)} , for image {image_path}")

    bubbles_array.extend(bubbles)

    return bubbles_array


def analyze_image(img, settings, filename=None, output_image=False):

    # Unpack settings
    SCALE_FACTOR = settings["SCALE_FACTOR"]
    SMALL_NOISE_FILTER_SIZE_DIAM = settings["SMALL_NOISE_FILTER_SIZE_DIAM"]  # Pixels
    SMALL_FILTER_SIZE_DIAM = settings["SMALL_FILTER_SIZE_DIAM"]  # Pixels
    GAUSSIAN_BLUR_SIZE = settings["GAUSSIAN_BLUR_SIZE"]
    ADAPTIVE_THRESHOLD_BLOCK_SIZE = settings["ADAPTIVE_THRESHOLD_BLOCK_SIZE"]
    ADAPTIVE_THRESHOLD_CONSTANT = settings["ADAPTIVE_THRESHOLD_CONSTANT"]
    MAX_MARKER_SIZE_FACTOR = settings["MAX_MARKER_SIZE_FACTOR"]
    MIN_SOLIDITY = settings["MIN_SOLIDITY"]
    MAX_ECCENTRICITY = settings["MAX_ECCENTRICITY"]

    if settings["CROP_AREA"] is not None:
        CROP_AREA = tuple(settings["CROP_AREA"])
    else:
        CROP_AREA = None

    # Check if a crop area is provided
    if CROP_AREA is not None:
        # Unpack the crop area
        x, y, w, h = CROP_AREA
        crop_img = img[y : y + h, x : x + w]
    else:
        # If no crop area is provided, use the whole image
        crop_img = img

    start = time.time()
    #### ---- Scale image to speed analysis ----####

    img_resize = cv2.resize(crop_img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    height, width, _ = img_resize.shape

    #### ---- Convert to grayscale ----####
    # Check if image is grayscale already
    if not len(img_resize.shape) == 2:
        # If shape isn't 2, the image is not grayscale, need to convert it
        gray_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img_resize

    # Convert to 8-bit format if it's not already
    gray_img = cv2.convertScaleAbs(gray_img)

    # # Plot the image
    # plt.imshow(gray_img, cmap="gray")
    # plt.axis("off")  # Hide axis labels
    # plt.show()

    #### ---- Apply Guassian blur ----####
    blur_img = cv2.GaussianBlur(gray_img, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)

    # # Plot the image
    # plt.imshow(blur_img, cmap="gray")
    # plt.axis("off")  # Hide axis labels
    # plt.show()

    #### ---- Apply auto threshold ----####
    # Set parameters for adaptive thresholding
    max_output_value = 255
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    threshold_type = cv2.THRESH_BINARY

    # Apply adaptive thresholding
    thresh_img = cv2.adaptiveThreshold(
        blur_img,
        max_output_value,
        adaptive_method,
        threshold_type,
        ADAPTIVE_THRESHOLD_BLOCK_SIZE,
        ADAPTIVE_THRESHOLD_CONSTANT,
    )

    # Invert trheshold image
    thresh_img = np.invert(thresh_img)
    thresh_img = thresh_img.astype(bool)

    # # Plot the image
    # plt.imshow(thresh_img, cmap="gray")
    # plt.axis("off")  # Hide axis labels
    # plt.show()

    #### ---- Remove noise from image ----####
    SMALL_FILTER_SIZE = SMALL_NOISE_FILTER_SIZE_DIAM**2 * np.pi / 4
    no_small_img = remove_small_objects(thresh_img, SMALL_FILTER_SIZE)

    # # Plot the image
    # plt.imshow(no_small_img, cmap="gray")
    # plt.axis("off")  # Hide axis labels
    # plt.show()

    # Slightly dilate
    # no_small_img = cv2.dilate(no_small_img.astype(np.uint8), None, iterations=1)

    inverse_img = np.invert(no_small_img)

    end = time.time()
    # print(f"Time to process image: {end - start}")

    #### ---- Extract markers ----####
    start = time.time()
    # Loop over each region
    start_rp = time.time()
    rp = regionprops(label(inverse_img))
    end_rp = time.time()
    # print(f"Time to perform regionprops: {end_rp - start_rp}")

    max_marker_size = height * width * MAX_MARKER_SIZE_FACTOR

    filtered_markers = []
    filtered_marker_image = np.zeros_like(inverse_img)

    for region in rp:
        # Get properties
        bubble_diam = region.equivalent_diameter
        solidity = region.solidity
        eccentricity = region.eccentricity

        # Apply filters
        if (
            bubble_diam <= max_marker_size
            # and solidity >= MIN_SOLIDITY
            and eccentricity <= MAX_ECCENTRICITY
        ):
            coords = region.coords
            label_ref = region.label
            for coord in coords:
                filtered_marker_image[coord[0], coord[1]] = label_ref

    # Plot the image
    # plt.imshow(filtered_marker_image, cmap="gray")
    # plt.axis("off")  # Hide axis labels
    # plt.title("filtered marker image")
    # plt.show()

    end = time.time()
    # print(f"Time to extract markers: {end - start}")

    start = time.time()

    #### ---- Create marker and watershed image ----####
    marker_img = np.copy(no_small_img)

    # Plot the image
    # plt.imshow(marker_img, cmap="gray")
    # plt.axis("off")  # Hide axis labels
    # plt.title("marker image")
    # plt.show()

    # Fill bubble holes with markers
    filled_img = np.logical_or(marker_img, filtered_marker_image).astype(int)

    # Fill remaining holes with scipy's ndimage
    filled_img = ndimage.binary_fill_holes(filled_img).astype(int)

    marker_img = marker_img != filled_img

    # Plot the image
    # plt.imshow(filled_img, cmap="gray")
    # plt.axis("off")  # Hide axis labels
    # plt.title("filled image")
    # plt.show()

    # Make it pretty
    # Create a color map for the marker_img
    cmap = (
        plt.cm.jet
    )  # You can use other colormaps like plt.cm.viridis, plt.cm.plasma, etc.
    normed_markers = marker_img / np.max(marker_img)
    colored_marker_img = (cmap(normed_markers)[:, :, :3] * 255).astype(np.uint8)

    # Overlay images
    overlay_img = np.where(
        marker_img[..., None].astype(bool),
        colored_marker_img,
        filled_img[..., None] * 255,
    )
    overlay_img = overlay_img.astype(np.uint8)

    # Plot the image
    # plt.imshow(overlay_img, cmap="gray")
    # plt.title("overlay image")
    # plt.axis("off")  # Hide axis labels
    # plt.show()

    # Plot the image
    # plt.imshow(filled_img, cmap="gray")
    # plt.axis("off")  # Hide axis labels
    # plt.title("filled image")
    # plt.show()

    #### ---- Perform watershed ----####
    # Ensure the image is in the correct format
    filled_img_uint8 = filled_img.astype(np.uint8)

    # Create sure background area with dilation
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(filled_img_uint8, kernel, iterations=2)

    # # Plot the image
    # plt.imshow(sure_bg, cmap="gray")
    # plt.axis("off")  # Hide axis labels
    # plt.show()

    marker_img_uint8 = marker_img.astype(np.uint8)

    # Create unknown region
    unknown = cv2.subtract(sure_bg, marker_img_uint8)

    # Create markers
    _, markers = cv2.connectedComponents(marker_img_uint8)

    # Make the background 1
    markers = markers + 1
    # Make the unknown region 0, where watershed should be applied
    markers[unknown == 1] = 0

    # Create distrance transform
    filled_img_uint8 = filled_img.astype(np.uint8)
    dist_transform = cv2.distanceTransform(filled_img_uint8, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(
        dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX
    )

    # Watershed
    markers = watershed(-dist_transform, markers)

    # Remove edge markers
    end = time.time()
    # print(f"Time to perform watershed: {end - start}")

    start = time.time()

    # Perform regionprops
    start_rp = time.time()
    rp = regionprops(markers)
    end_rp = time.time()

    # print(f"Time to perform regionprops: {end_rp - start_rp}")

    # Perform filtering here
    min_bubble_diam = SMALL_FILTER_SIZE_DIAM

    rp = rp[1:]  # Get rid of background label

    # Filter regions based on your criteria and store equivalent diameters
    filtered_regions = []
    for r in rp:
        if (
            r.equivalent_diameter > min_bubble_diam
            and r.solidity > MIN_SOLIDITY
            and r.eccentricity < MAX_ECCENTRICITY
        ):
            filtered_regions.append(r)

    # Create a new image to store filtered labels
    filtered_markers = np.zeros_like(markers)

    # Assign new labels to the filtered regions
    for region in filtered_regions:
        filtered_markers[markers == region.label] = region.label

    filtered_markers = clear_border(filtered_markers)

    rp = regionprops(filtered_markers)
    equiv_diameters = [r.equivalent_diameter for r in rp]

    end = time.time()
    # print(f"Time to filter regionprops: {end - start}")

    #### ---- Output image ----####

    if output_image:
        start = time.time()

        image_label_overlay = label2rgb(filtered_markers, image=gray_img, bg_label=0)
        image_label_overlay = (image_label_overlay * 255).astype("uint8")

        # Save image
        cv2.imwrite(filename, image_label_overlay)
        print(f"Image saved to {filename}")
        end = time.time()
        # print(f"Time to output image: {end - start}")

    return equiv_diameters


def remove_outliers(values):
    # Create a Series from the list
    s = pd.Series(values)

    # Calculate Q1, Q3 and IQR
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1

    # Define the upper and lower bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create a mask for values within the bounds
    mask = (s >= lower_bound) & (s <= upper_bound)

    # Apply the mask to get the Series without outliers
    filtered_values = s[mask]

    # Create a mask for values outside the bounds (outliers)
    outlier_mask = (s < lower_bound) | (s > upper_bound)

    # Apply the mask to get the Series with only outliers
    outliers = s[outlier_mask]

    # Convert the Series back to a list
    filtered_values_list = filtered_values.tolist()
    outliers_list = outliers.tolist()

    return filtered_values_list, outliers_list


def summarize_bubble_data(bubbles_list):
    """
    Calculate summary data to output to SensorMeasurement object.
    """
    summary = {
        "histogram_data": calculate_histogram_bins(bubbles_list, bins="auto"),
        "d32": sauter_mean_diameter(bubbles_list),
        "d10": np.mean(bubbles_list),
        "std": np.std(bubbles_list),
        "n": len(bubbles_list),
    }
    return summary


def calculate_histogram_bins(data, bins="auto"):
    """
    Calculate histogram bin counts and edges.

    Parameters:
        data (list of float): The input data for which histogram bins are calculated.
        bins (int, sequence of scalars, or str, optional): The method used to calculate the bins.
            - 'auto' (default): Automatically determine the best bin size using numpy's algorithm.
            - int: Specify the number of bins.
            - sequence: Define the bin edges explicitly.

    Returns:
        dict: A dictionary containing the bin edges and counts.
    """
    # Calculate the bin edges and bin counts
    bin_edges = np.histogram_bin_edges(data, bins=bins)
    counts, _ = np.histogram(data, bins=bin_edges)
    frequency, _ = np.histogram(data, bins=bin_edges, density=True)

    # Convert frequency to a more intuitive percentage by multiplying by bin width
    bin_widths = np.diff(bin_edges)
    frequency_percentage = frequency * bin_widths  # frequency in percentage

    # Prepare the output dictionary
    histogram_data = {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "frequency": frequency_percentage.tolist(),
    }

    return histogram_data

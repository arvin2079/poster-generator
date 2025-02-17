import sys
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display


def imshow_in_screen_size(winname: str, image: np.ndarray):
    # Get the screen size using tkinter
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.quit()

    # Limit the window height to 95% of the screen height
    max_window_height = int(screen_height * 0.85)

    # Calculate the original aspect ratio of the image
    height, width = image.shape[:2]
    aspect_ratio = width / height

    # Resize the window if the image is larger than the screen size or 95% screen height
    if width > screen_width or height > max_window_height:
        if width > screen_width:
            width = screen_width
            height = int(screen_width / aspect_ratio)
        if height > max_window_height:
            height = max_window_height
            width = int(max_window_height * aspect_ratio)

    # Resize the image to fit within the screen
    resized_image = cv2.resize(image, (width, height))

    cv2.imshow(winname, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_quadrilateral(image: str | np.ndarray, points: np.ndarray) -> np.ndarray:
    # Load the image if the input is a path
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # Ensure the points array has the right shape (4, 2)
    if points.shape != (4, 2):
        raise ValueError("Points array must be of shape (4, 2).")

    # Draw the quadrilateral
    pts = points.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=8)

    return image


def draw_quadrilateral_with_label(
    image: str | np.ndarray, points: np.ndarray, label: str
) -> np.ndarray:
    # Load the image if the input is a path
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # Ensure the points array has the right shape (4, 2)
    if points.shape != (4, 2):
        raise ValueError("Points array must be of shape (4, 2).")

    # Draw the quadrilateral
    pts = points.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=8)

    # Calculate the centroid of the quadrilateral (average of the points)
    centroid_x = int(np.mean(points[:, 0]))
    centroid_y = int(np.mean(points[:, 1]))

    # Define the font and scale for the label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (0, 0, 0)  # White color for the text
    font_thickness = 3

    # Get the text size to center it
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = centroid_x - text_size[0] // 2
    text_y = centroid_y + text_size[1] // 2

    # Put the text label at the centroid
    cv2.putText(
        image, label, (text_x, text_y), font, font_scale, font_color, font_thickness
    )

    return image


def remove_background(
    image: str | np.ndarray, output_path: Optional[str] = None, threshold_value=150
):
    """
    Removes the background from an image containing handwritten text and saves the result as a PNG with transparency.

    Parameters:
    - image_path (str or std numpy array of the image): The path to the input image (supports both JPG and PNG).
    - output_path (str): The path to save the output PNG image with a transparent background. Default is 'handwritten_text.png'.
    - threshold_value (int): The threshold value for binary segmentation (default is 150). Adjust based on contrast of text and background.

    Returns:
    - None or numpy.ndarray
    """

    # Load the image with transparency if it's PNG or without it (for JPG or non-transparent images)
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # Check if the image has 4 channels (for PNGs with transparency)
    if image.shape[2] == 4:
        # Separate the image into BGR and alpha channels
        b, g, r, alpha = cv2.split(image)

        # Convert BGR to grayscale for thresholding
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        # If the image doesn't have an alpha channel, assume it's BGR (3 channels)
        # Split the BGR channels
        b, g, r = cv2.split(image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a synthetic alpha channel (255 for fully opaque)
        alpha = np.ones(gray.shape, dtype=gray.dtype) * 255

    # Apply a binary threshold to create a mask for the background
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Update the alpha channel using the mask (make background transparent)
    alpha = cv2.bitwise_and(alpha, mask)

    # Merge the BGR channels (or RGB in OpenCV terms) with the updated alpha channel
    rgba = cv2.merge([b, g, r, alpha])

    # Save the result as a PNG file with transparency
    if output_path:
        cv2.imwrite(output_path, rgba)

    return rgba


def overlay_image_with_perspective(
    base_image: str | np.ndarray,
    overlay_image: str | np.ndarray,
    dst_points: np.ndarray,
    output_path: Optional[str] = None,
):
    """
    Overlays one image onto another using perspective transformation defined by four points.

    :param base_image: Path to the base image (background image) or numpy array.
    :param overlay_image: Path to the overlay image (foreground image) or numpy array.
    :param dst_points: Array of points defining where to place the overlay image in the base image (shape: (4, 2)).
    :param output_path: Path to save the resulting image.
    """

    def resize_and_center_quadrilateral(points, target_aspect_ratio):
        """
        Resizes the quadrilateral to the target aspect ratio while maintaining its shape
        and centers the resized quadrilateral inside the original one.

        Parameters:
        - points: ndarray of shape (4, 2), containing the coordinates of the quadrilateral vertices
        - target_aspect_ratio: desired width/height ratio for the resized quadrilateral

        Returns:
        - new_points: the resized and centered quadrilateral points
        """
        # Calculate width and height of the original quadrilateral
        top_left, top_right, bottom_right, bottom_left = points

        # Calculate the width and height of the original quadrilateral
        original_width = np.linalg.norm(top_right - top_left)
        original_height = np.linalg.norm(bottom_left - top_left)

        # Calculate new dimensions based on the target aspect ratio
        new_height = min(original_height, original_width / target_aspect_ratio)
        new_width = new_height * target_aspect_ratio

        # Calculate scaling factors for the quadrilateral
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        # Compute the center of the original quadrilateral
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        # Scale the points to match the new aspect ratio
        scaled_points = []
        for pt in points:
            # Translate points so the center is at the origin
            translated_pt = pt - np.array([center_x, center_y])
            
            # Scale points
            scaled_pt = translated_pt * np.array([scale_x, scale_y])
            
            # Translate points back to the original center
            scaled_points.append(scaled_pt + np.array([center_x, center_y]))

        # Convert the result to a numpy array
        new_points = np.array(scaled_points, dtype=np.int32)

        return new_points

    def add_alpha_channel(image: np.ndarray) -> np.ndarray:
        """
        Adds an alpha channel to an image if it doesn't already have one.

        :param image: The input image (numpy array).
        :return: Image with alpha channel (numpy array).
        """
        if image.shape[2] == 3:  # If image has no alpha channel (i.e., only BGR)
            # Create a new image with an additional alpha channel
            bgr = image[:, :, :3]  # Extract BGR channels
            alpha_channel = (
                np.ones(bgr.shape[:2], dtype=bgr.dtype) * 255
            )  # Full opacity
            image_with_alpha = np.dstack([bgr, alpha_channel])  # Add the alpha channel
            return image_with_alpha
        return image  # Return unchanged if alpha channel is present

    # Load images if provided as file paths
    if isinstance(base_image, str):
        base_image = cv2.imread(base_image, cv2.IMREAD_UNCHANGED)
    if isinstance(overlay_image, str):
        overlay_image = cv2.imread(overlay_image, cv2.IMREAD_UNCHANGED)

    # Ensure both images are loaded correctly
    if base_image is None or overlay_image is None:
        raise ValueError("Could not load base or overlay image.")

    # Ensure base image has 4 channels (BGRA) to handle transparency
    if base_image.shape[2] != 4:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2BGRA)

    # Add alpha channel to overlay image if necessary
    overlay_image = add_alpha_channel(overlay_image)

    # Get dimensions of the overlay image
    overlay_height, overlay_width = overlay_image.shape[:2]

    # overlay image aspect ratio
    overlay_aspect_ratio = overlay_width / overlay_height

    # resize the dst_points to maintain the overlay image shape
    dst_points = resize_and_center_quadrilateral(dst_points, overlay_aspect_ratio)

    # Define source points based on overlay image size
    src_points = np.float32(
        [
            [0, 0],
            [overlay_width, 0],
            [overlay_width, overlay_height],
            [0, overlay_height],
        ]
    )

    # Convert dst_points to float32
    dst_points = np.float32(dst_points)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the overlay image (with alpha channel) using the perspective matrix
    transformed_overlay = cv2.warpPerspective(
        overlay_image,
        matrix,
        (base_image.shape[1], base_image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    # Separate the alpha channel from the transformed overlay
    if transformed_overlay.shape[2] == 4:
        overlay_alpha = (
            transformed_overlay[:, :, 3] / 255.0
        )  # Normalize alpha to [0, 1]
        overlay_rgb = transformed_overlay[:, :, :3]
    else:
        raise ValueError("Overlay image does not have an alpha channel.")

    # Separate base image into RGB and alpha channels
    base_rgb = base_image[:, :, :3]
    base_alpha = base_image[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

    # Blend the images based on the alpha channel
    combined_rgb = (
        overlay_rgb * overlay_alpha[:, :, None]
        + base_rgb * (1 - overlay_alpha[:, :, None])
    ).astype(np.uint8)

    # Update the alpha channel in the base image where the overlay is applied
    combined_alpha = ((overlay_alpha + base_alpha * (1 - overlay_alpha)) * 255).astype(
        np.uint8
    )

    # Combine RGB and alpha channels back into a single image
    combined_image = np.dstack([combined_rgb, combined_alpha])

    # Save or return the resulting image
    if output_path:
        cv2.imwrite(output_path, combined_image)

    return combined_image


def project_text_on_image(
    image: str | np.ndarray,
    text,
    output_path=None,
    points=None,
    font_path=None,
    font_size=32,
    word_spacing=20,
    line_spacing=10,
    text_alignment="justified",
):
    # Load the image using OpenCV
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # Convert the OpenCV image (BGR) to a PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Load the custom .ttf font using PIL
    if font_path is None:
        raise ValueError(
            "A valid font path (.ttf) is required for custom font rendering."
        )

    # Reshape the Persian text for correct letter shaping and bidi
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    # Extract rectangle's four points (top-left, top-right, bottom-left, bottom-right)
    if len(points) != 4:
        raise ValueError("Four points are required to define the rectangle.")

    x1, y1 = points[0]  # Top-left corner
    x2, y2 = points[1]  # Top-right corner
    x3, y3 = points[2]  # Bottom-left corner
    x4, y4 = points[3]  # Bottom-right corner

    # Calculate the width and height of the rectangle
    rect_width = x2 - x1
    rect_height = y3 - y1

    # Split the text into words and process $$ for green coloring
    words = bidi_text.split()
    green_words = []
    processed_words = []

    # Process text to handle $$ for green words
    for word in words:
        # if getattr(sys, 'frozen', False):
        #     word = word[::-1]
        # else:
        #     pass
        if "$$" in word:
            # Remove $$ and add to green words list
            clean_word = word.replace("$$", "")
            green_words.append(clean_word)
            processed_words.append((clean_word, "green"))
        else:
            processed_words.append((word, "black"))

    # Reverse the order of words before writing to the image
    if text_alignment == 'justified': 
        processed_words.reverse()  # Reverse word order here

    # Try different font sizes until the text fits within the rectangle
    def fits_in_box(font_size):
        # Load the font with the given size
        font = ImageFont.truetype(font_path, font_size)
        current_x, current_y = (
            x2,
            y1,
        )  # Start from the top-right corner of the rectangle
        line_height = 0  # To track the height of the current line of text

        for word, color in processed_words:
            # Get the size of the word using textbbox
            text_bbox = draw.textbbox((0, 0), word, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Check if the word fits within the rectangle's width (x1 to x2)
            if current_x - text_width < x1:
                # If the word doesn't fit, move to the next line (within the rectangle)
                current_x = x2  # Reset x to the right side of the rectangle
                current_y += line_height + line_spacing  # Move y down
                line_height = text_height  # Set new line height

            # Check if the text fits within the rectangle's height (y1 to y3)
            if current_y + line_height > y3:
                return False  # Return False if the text exceeds the bottom of the rectangle

            # Update the x position to move leftward with the word's width and spacing
            current_x -= text_width + word_spacing
            line_height = max(line_height, text_height)

        return True  # Return True if the text fits inside the rectangle

    # Try reducing the font size until it fits
    while (
        not fits_in_box(font_size) and font_size > 10
    ):  # Avoid font size going too small
        font_size -= 1  # Decrease font size until it fits

    # Now that we have the appropriate font size, draw the text on the image
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image_pil)

    # Starting point for the text (top-right corner of the rectangle)
    current_x, current_y = x2, y1
    line_height = 0

    # To store all lines of words
    lines = []
    current_line_words = []

    # Draw the text and split into lines
    for word, color in processed_words:
        text_bbox = draw.textbbox((0, 0), word, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if current_x - text_width < x1:
            # Move to the next line
            lines.append(current_line_words)
            current_x = x2
            current_y += line_height + line_spacing
            line_height = text_height
            current_line_words = []

        if current_y + line_height > y3:
            break  # Stop drawing if text exceeds the bottom of the rectangle

        current_line_words.append((word, color))
        current_x -= text_width + word_spacing
        line_height = max(line_height, text_height)

    if current_line_words:
        lines.append(current_line_words)

    # Calculate total height of the text block for vertical centering
    total_text_height = sum(
        max(
            draw.textbbox((0, 0), word, font=font)[3]
            - draw.textbbox((0, 0), word, font=font)[1]
            for word, color in line
        )
        + line_spacing
        for line in lines
    )
    total_text_height -= line_spacing  # Remove extra line spacing after the last line

    # Adjust font size if there are more than 3 lines
    if len(lines) > 3:
        font_size -= 2  # Decrease font size to fit within 3 lines (adjust the value as needed)
    
    # Re-calculate the text layout with the adjusted font size
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image_pil)
    current_x, current_y = x2, y1
    lines = []
    current_line_words = []

    # Draw the text and split into lines again after adjusting font size
    for word, color in processed_words:
        text_bbox = draw.textbbox((0, 0), word, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if current_x - text_width < x1:
            lines.append(current_line_words)
            current_x = x2
            current_y += line_height + line_spacing
            line_height = text_height
            current_line_words = []

        if current_y + line_height > y3:
            break

        current_line_words.append((word, color))
        current_x -= text_width + word_spacing
        line_height = max(line_height, text_height)

    if current_line_words:
        lines.append(current_line_words)

    # Calculate starting y position to vertically center the text
    total_text_height = sum(
        max(
            draw.textbbox((0, 0), word, font=font)[3]
            - draw.textbbox((0, 0), word, font=font)[1]
            for word, color in line
        )
        + line_spacing
        for line in lines
    )
    total_text_height -= line_spacing

    current_y = y1 + (rect_height - total_text_height) // 2

    # Draw the text based on the selected alignment
    lines.reverse()
    for line in lines:
        total_width = sum(
            draw.textbbox((0, 0), word, font=font)[2]
            - draw.textbbox((0, 0), word, font=font)[0]
            for word, color in line
        )
        line_height = max(
            draw.textbbox((0, 0), word, font=font)[3]
            - draw.textbbox((0, 0), word, font=font)[1]
            for word, color in line
        )

        if text_alignment == "justified":
            remaining_space = rect_width - total_width
            total_spaces = len(line) - 1
            if total_spaces > 0 and line != lines[-1]:
                space_between_words = remaining_space // total_spaces
            else:
                space_between_words = word_spacing

            current_x = x2
            for word, color in line:
                text_bbox = draw.textbbox((0, 0), word, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                draw.text(
                    (current_x - text_width, current_y),
                    word,
                    font=font,
                    fill=color,
                )
                current_x -= text_width + space_between_words

        elif text_alignment == "centered":
            # Calculate starting x to center the line
            total_line_width = sum(
                draw.textbbox((0, 0), word, font=font)[2]
                - draw.textbbox((0, 0), word, font=font)[0]
                for word, color in line
            )
            start_x = x1 + (rect_width - total_line_width) // 2
            current_x = start_x
            for word, color in line:
                text_bbox = draw.textbbox((0, 0), word, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                draw.text(
                    (current_x, current_y),
                    word,
                    font=font,
                    fill=color,
                )
                current_x += text_width + word_spacing - 20

        current_y += line_height + line_spacing

    # Save the image
    image_pil = image_pil.convert("RGB")
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    if output_path:
        cv2.imwrite(output_path, image)

    return image


def project_text_on_image_two_line(
    image: str | np.ndarray,
    text,
    output_path=None,
    points=None,
    font_path=None,
    line_spacing=10,
    text_alignment="justified",
):
    # Load the image using OpenCV
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # Convert the OpenCV image (BGR) to a PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Load the custom .ttf font using PIL
    if font_path is None:
        raise ValueError(
            "A valid font path (.ttf) is required for custom font rendering."
        )

    # Reshape the Persian text for correct letter shaping and bidi
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    # Extract rectangle's four points (top-left, top-right, bottom-left, bottom-right)
    if len(points) != 4:
        raise ValueError("Four points are required to define the rectangle.")

    x1, y1 = points[0]  # Top-left corner
    x2, y2 = points[1]  # Top-right corner
    x3, y3 = points[2]  # Bottom-left corner
    x4, y4 = points[3]  # Bottom-right corner

    # Calculate the width and height of the rectangle
    rect_width = x2 - x1
    rect_height = y3 - y1

    # Split the text into words and process $$ for green coloring
    words = bidi_text.split()
    green_words = []
    processed_words = []

    # Process text to handle $$ for green words
    for word in words:
        # if getattr(sys, 'frozen', False):
        #     word = word[::-1]
        # else:
        #     pass

        if "$$" in word:
            # Remove $$ and add to green words list
            clean_word = word.replace("$$", "")
            green_words.append(clean_word)
            processed_words.append((clean_word, "green"))
        else:
            processed_words.append((word, "black"))

    # Reverse the order of words before writing to the image
    processed_words.reverse()  # Reverse word order here

    # Define a function to check if the text fits in two lines at a given font size and word spacing
    def fits_in_two_lines(font_size, word_spacing):
        # Load the font with the given size
        
        font = ImageFont.truetype(font_path, font_size)
        current_x, current_y = (
            x2,
            y1,
        )  # Start from the top-right corner of the rectangle
        line_height = 0  # To track the height of the current line of text
        lines = 0  # To track the number of lines

        for word, color in processed_words:
            # Get the size of the word using textbbox
            text_bbox = draw.textbbox((0, 0), word, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if current_x - text_width < x1:
                # Move to the next line
                current_x = x2
                current_y += line_height + line_spacing
                line_height = text_height
                lines += 1

                # If more than 2 lines, return False
                if lines >= 2:
                    return False

            # Update the x position to move leftward with the word's width and spacing
            current_x -= text_width + word_spacing
            line_height = max(line_height, text_height)

        # Ensure text fits within exactly two lines
        return lines == 1  # Returns True if the text fits exactly into two lines

    # Try to find the largest font size and word spacing that fits the text into two lines
    font_size = 100  # Start with a large font size
    word_spacing = 20  # Start with the minimum word spacing

    while not fits_in_two_lines(font_size, word_spacing) and font_size > 10:
        # Adjust word spacing based on the font size and coefficient
        if font_size < 60:
            word_spacing = 20
        else:
            word_spacing = font_size * 0.5 + 20

        font_size -= 1  # Decrease font size until it fits into two lines

    # Now that we have the appropriate font size and word spacing, draw the text on the image
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image_pil)

    # Starting point for the text (top-right corner of the rectangle)
    current_x, current_y = x2, y1
    line_height = 0

    # To store all lines of words
    lines = []
    current_line_words = []

    # Split text into two lines
    for word, color in processed_words:
        text_bbox = draw.textbbox((0, 0), word, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if current_x - text_width < x1:
            # Move to the next line
            lines.append(current_line_words)
            current_x = x2
            current_y += line_height + line_spacing
            line_height = text_height
            current_line_words = []

        current_line_words.append((word, color))
        current_x -= text_width + word_spacing
        line_height = max(line_height, text_height)

    if current_line_words:
        lines.append(current_line_words)

    # Calculate total height of the text block for vertical centering
    total_text_height = sum(
        max(
            draw.textbbox((0, 0), word, font=font)[3]
            - draw.textbbox((0, 0), word, font=font)[1]
            for word, color in line
        )
        + line_spacing
        for line in lines
    )
    total_text_height -= line_spacing  # Remove extra line spacing after the last line

    # Calculate starting y position to vertically center the text
    current_y = (
        y1 + (rect_height - total_text_height) // 2 - (15 if font_size < 60 else 20)
    )

    # Draw the text based on the selected alignment
    for line in lines:
        total_width = sum(
            draw.textbbox((0, 0), word, font=font)[2]
            - draw.textbbox((0, 0), word, font=font)[0]
            for word, color in line
        )
        line_height = max(
            draw.textbbox((0, 0), word, font=font)[3]
            - draw.textbbox((0, 0), word, font=font)[1]
            for word, color in line
        )

        if text_alignment == "justified":
            # Fully justified right-to-left text
            # for line in lines:
            total_width = sum(
                draw.textbbox((0, 0), word, font=font)[2]
                - draw.textbbox((0, 0), word, font=font)[0]
                for word, color in line
            )
            remaining_space = rect_width - total_width
            total_spaces = len(line) - 1
            
            if total_spaces > 0:  # Fully justify both lines
                # Calculate the exact space to add between words
                space_between_words = remaining_space / total_spaces
            else:
                space_between_words = word_spacing  # Default word spacing if only one word

            current_x = x2  # Start from the right edge
            for index, (word, color) in enumerate(line):
                text_bbox = draw.textbbox((0, 0), word, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                
                # Draw the word only once
                draw.text(
                    (current_x - text_width, current_y),
                    word,
                    font=font,
                    fill=(46, 147, 70) if color == "green" else (0, 0, 0),
                )
                
                # Move to the next word position
                current_x -= text_width + space_between_words  # Apply the calculated spacing

            # current_y += line_height + line_spacing  # Move to the next line

        elif text_alignment == "centered":
            # Centered text alignment
            remaining_space = rect_width - total_width
            current_x = x1 + remaining_space // 2
            for word, color in line:
                text_bbox = draw.textbbox((0, 0), word, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                draw.text(
                    (current_x, current_y),
                    word,
                    font=font,
                    fill=(46, 147, 70) if color == "green" else (0, 0, 0),
                )
                current_x += text_width + word_spacing

        current_y += line_height + line_spacing

    # Convert the PIL image back to OpenCV format (BGR)
    final_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # If an output path is provided, save the final image
    if isinstance(output_path, str):
        cv2.imwrite(output_path, final_image)

    # Return the final image as a NumPy array
    return final_image


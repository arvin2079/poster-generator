import cv2
import numpy as np
import tkinter as tk


# def show_image_in_opencv(image):
#     # Display image using OpenCV's imshow
#     cv2.imshow("OpenCV Image", image)

#     # Wait for the user to close the OpenCV window (non-blocking mode)
#     while cv2.getWindowProperty("OpenCV Image", cv2.WND_PROP_VISIBLE) >= 1:
#         key = cv2.waitKey(100)
#         if key == 27:  # ESC key to close
#             break
#     cv2.destroyWindow("OpenCV Image")

# def open_imshow_in_thread(image_path):
#     # Start the OpenCV imshow in a separate thread so it doesn't block the Tkinter GUI
#     threading.Thread(target=show_image_in_opencv, args=(image_path,), daemon=True).start()


def select_rectangle(image: str | np.ndarray):
    # Load the image
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

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

    # Add a box with text at the top center of the image
    text = "Click to select one corner of a rectangle"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # White color for the text
    thickness = 2

    # Get text size to position the box at the top center
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    box_x1 = (width - text_width) // 2 - 10  # 10px padding
    box_x2 = box_x1 + text_width + 20  # 10px padding on each side
    box_y1 = 10
    box_y2 = box_y1 + text_height + 10  # Extra padding for the box height

    # Draw the box
    cv2.rectangle(
        resized_image, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1
    )  # Black box
    # Put text inside the box
    cv2.putText(
        resized_image,
        text,
        (box_x1 + 10, box_y1 + text_height + 5),
        font,
        font_scale,
        font_color,
        thickness,
    )

    # Variable to store the starting corner
    start_corner = []

    # Mouse callback function to capture the rectangle selection
    def click_and_capture_rectangle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not start_corner:
                start_corner.append((x, y))
                cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("OpenCV Image", resized_image)

    # Display the image and allow the user to select the rectangle's first corner
    cv2.imshow("OpenCV Image", resized_image)
    cv2.setMouseCallback("OpenCV Image", click_and_capture_rectangle)

    # Wait until the start corner is selected or the window is closed
    while not start_corner:
        if cv2.getWindowProperty("OpenCV Image", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None
        cv2.waitKey(1)

    # Let the user define the size of the rectangle interactively
    def draw_rectangle(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and start_corner:
            # Get the starting corner
            x1, y1 = start_corner[0]

            # Draw a rectangle as the user moves the mouse
            temp_image = resized_image.copy()
            cv2.rectangle(temp_image, (x1, y1), (x, y), (0, 255, 0), 2)
            cv2.imshow("OpenCV Image", temp_image)

        if event == cv2.EVENT_LBUTTONDOWN and start_corner:
            # Finish and confirm the rectangle selection
            x1, y1 = start_corner[0]

            # Draw the final rectangle
            cv2.rectangle(resized_image, (x1, y1), (x, y), (0, 255, 0), 2)
            cv2.imshow("OpenCV Image", resized_image)

            # Store both corners as points
            start_corner.append((x, y))

    cv2.setMouseCallback("OpenCV Image", draw_rectangle)

    # Wait for the user to finalize the rectangle selection
    while len(start_corner) < 2:
        if cv2.getWindowProperty("OpenCV Image", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None
        cv2.waitKey(1)

    # Calculate the scaling factors
    scaling_factor_x = image.shape[1] / float(resized_image.shape[1])
    scaling_factor_y = image.shape[0] / float(resized_image.shape[0])

    # Convert the selected rectangle back to the original image's coordinates
    x1, y1 = start_corner[0]
    x2, y2 = start_corner[1]

    # Scale the coordinates back to the original image size
    original_corners = np.array(
        [
            [int(x1 * scaling_factor_x), int(y1 * scaling_factor_y)],
            [int(x2 * scaling_factor_x), int(y1 * scaling_factor_y)],
            [int(x2 * scaling_factor_x), int(y2 * scaling_factor_y)],
            [int(x1 * scaling_factor_x), int(y2 * scaling_factor_y)],
        ]
    )

    # Close the OpenCV window
    cv2.destroyAllWindows()

    return original_corners


def select_square(image: str | np.ndarray):
    # Load the image
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

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

    # Add a box with text at the top center of the image
    text = "Click to select one corner of a square"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # White color for the text
    thickness = 2

    # Get text size to position the box at the top center
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    box_x1 = (width - text_width) // 2 - 10  # 10px padding
    box_x2 = box_x1 + text_width + 20  # 10px padding on each side
    box_y1 = 10
    box_y2 = box_y1 + text_height + 10  # Extra padding for the box height

    # Draw the box
    cv2.rectangle(
        resized_image, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1
    )  # Black box
    # Put text inside the box
    cv2.putText(
        resized_image,
        text,
        (box_x1 + 10, box_y1 + text_height + 5),
        font,
        font_scale,
        font_color,
        thickness,
    )

    # Variable to store the starting corner
    start_corner = []

    # Mouse callback function to capture the square selection
    def click_and_capture_square(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not start_corner:
                start_corner.append((x, y))
                cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("OpenCV Image", resized_image)

    # Display the image and allow the user to select the square's first corner
    cv2.imshow("OpenCV Image", resized_image)
    cv2.setMouseCallback("OpenCV Image", click_and_capture_square)

    # Wait until the start corner is selected or the window is closed
    while not start_corner:
        if cv2.getWindowProperty("OpenCV Image", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None
        cv2.waitKey(1)

    # Let the user define the size of the square interactively
    def draw_square(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and start_corner:
            # Calculate the size of the square based on the diagonal distance
            x1, y1 = start_corner[0]
            side_length = min(abs(x - x1), abs(y - y1))  # Ensure square shape

            # Compute the second corner of the square
            x2 = x1 + side_length if x > x1 else x1 - side_length
            y2 = y1 + side_length if y > y1 else y1 - side_length

            temp_image = resized_image.copy()
            cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("OpenCV Image", temp_image)

        if event == cv2.EVENT_LBUTTONDOWN and start_corner:
            # Finish and confirm the square selection
            x1, y1 = start_corner[0]
            side_length = min(abs(x - x1), abs(y - y1))
            x2 = x1 + side_length if x > x1 else x1 - side_length
            y2 = y1 + side_length if y > y1 else y1 - side_length

            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("OpenCV Image", resized_image)

            # Store both corners as points
            start_corner.append((x2, y2))

    cv2.setMouseCallback("OpenCV Image", draw_square)

    # Wait for the user to finalize the square selection
    while len(start_corner) < 2:
        if cv2.getWindowProperty("OpenCV Image", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None
        cv2.waitKey(1)

    # Calculate the scaling factors
    scaling_factor_x = image.shape[1] / float(resized_image.shape[1])
    scaling_factor_y = image.shape[0] / float(resized_image.shape[0])

    # Convert the selected square back to the original image's coordinates
    x1, y1 = start_corner[0]
    x2, y2 = start_corner[1]

    # Calculate the four corners of the square
    if x2 > x1 and y2 > y1:
        top_left = (x1, y1)
        top_right = (x2, y1)
        bottom_right = (x2, y2)
        bottom_left = (x1, y2)
    elif x2 < x1 and y2 > y1:
        top_left = (x2, y1)
        top_right = (x1, y1)
        bottom_right = (x1, y2)
        bottom_left = (x2, y2)
    elif x2 < x1 and y2 < y1:
        top_left = (x2, y2)
        top_right = (x1, y2)
        bottom_right = (x1, y1)
        bottom_left = (x2, y1)
    else:
        top_left = (x1, y2)
        top_right = (x2, y2)
        bottom_right = (x2, y1)
        bottom_left = (x1, y1)

    # Scale the coordinates back to the original image size
    original_corners = np.array(
        [
            [int(top_left[0] * scaling_factor_x), int(top_left[1] * scaling_factor_y)],
            [
                int(top_right[0] * scaling_factor_x),
                int(top_right[1] * scaling_factor_y),
            ],
            [
                int(bottom_right[0] * scaling_factor_x),
                int(bottom_right[1] * scaling_factor_y),
            ],
            [
                int(bottom_left[0] * scaling_factor_x),
                int(bottom_left[1] * scaling_factor_y),
            ],
        ]
    )

    # Close the OpenCV window
    cv2.destroyAllWindows()

    return original_corners


def select_four_points(image: str | np.ndarray):
    # Load the image
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

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

    # Add a box with text at the top center of the image
    text = "Click to select four points"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # White color for the text
    thickness = 2

    # Get text size to position the box at the top center
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    box_x1 = (width - text_width) // 2 - 10  # 10px padding
    box_x2 = box_x1 + text_width + 20  # 10px padding on each side
    box_y1 = 10
    box_y2 = box_y1 + text_height + 10  # Extra padding for the box height

    # Draw the box
    cv2.rectangle(
        resized_image, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1
    )  # Black box
    # Put text inside the box
    cv2.putText(
        resized_image,
        text,
        (box_x1 + 10, box_y1 + text_height + 5),
        font,
        font_scale,
        font_color,
        thickness,
    )

    # List to store the points selected by the user
    points = []

    # Mouse callback function to capture the four points
    def click_and_capture_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Capture the point when clicked
            if len(points) < 4:  # We only want to capture 4 points
                points.append((x, y))
                cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("OpenCV Image", resized_image)

    # Display the image and allow the user to select four points
    cv2.imshow("OpenCV Image", resized_image)
    cv2.setMouseCallback("OpenCV Image", click_and_capture_points)

    # Wait until 4 points are selected or the window is closed
    while len(points) < 4:
        # Check if the window is closed
        if cv2.getWindowProperty("OpenCV Image", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None
        cv2.waitKey(1)

    # Calculate the scaling factors
    scaling_factor_x = image.shape[1] / float(resized_image.shape[1])
    scaling_factor_y = image.shape[0] / float(resized_image.shape[0])

    # Convert the selected points back to the original image's coordinates
    original_points = np.array(
        [[int(x * scaling_factor_x), int(y * scaling_factor_y)] for x, y in points]
    )

    # Close the OpenCV window
    cv2.destroyAllWindows()

    return original_points

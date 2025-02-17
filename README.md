# Photo Editing & Poster Generator Automation

## Overview
This project is a Python-based automation tool for editing photos and generating posters. It supports applying text (with or without perspective), adding images, and embedding QR codes onto predefined poster templates. The tool simplifies the process of customizing posters with dynamic content while ensuring consistency and efficiency.

## Features
- **Poster Template Management**: Load and use predefined poster templates.
- **Text Customization**: Add text with adjustable position, size, font, and optional perspective transformations.
- **Image Insertion**: Overlay images onto posters with precise positioning.
- **QR Code Embedding**: Generate and place QR codes on posters.
- **Automated Processing**: Streamlined workflow for batch poster generation.

## Installation
### Requirements
- Python 3.x
- Required dependencies (install via pip):
  ```bash
  pip install -r requirements.txt
  ```

### Dependencies
This project requires the following Python libraries:
- `Pillow` (for image manipulation)
- `OpenCV` (for perspective transformations)
- `numpy` (for mathematical operations)

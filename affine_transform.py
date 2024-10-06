import cv2
import torch
import numpy as np
from torch import Tensor

def tensor2cv2(image: Tensor) -> np.ndarray:
    img = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0)) 
    return img

def cv22tensor(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    if image.ndim == 3 and image.shape[0] == 3:
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    else:
        tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor

class AffineTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rect_x1": ("INT",),
                "rect_y1": ("INT",),
                "rect_x2": ("INT",),
                "rect_y2": ("INT",),
                "rect_x3": ("INT",),
                "rect_y3": ("INT",),
                "rect_x4": ("INT",),
                "rect_y4": ("INT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform"

    CATEGORY = "image/"

    def transform(self, image, rect_x1, rect_y1, rect_x2, rect_y2, rect_x3, rect_y3, rect_x4, rect_y4):
        image = tensor2cv2(image)

        # x1, y1 = 0, 0  # Top-left corner of the ROI
        # x2, y2 = image.shape[:2]  # Bottom-right corner of the ROI
        # roi = image[y1:y2, x1:x2]

        # # Resize the selected part (ROI)
        # new_size = (zoom_rate * x2, zoom_rate * y2)  # New width and height for the ROI
        # resized_roi = cv2.resize(roi, new_size)

        # # Create a mask for the resized part
        # mask = np.zeros_like(image)
        # mask[y1:y1+new_size[1], x1:x1+new_size[0]] = resized_roi

        # # Define the translation matrix to move the resized part to a new location
        # dx, dy = pos_x, pos_y  # Move the resized part to (x1 + dx, y1 + dy)
        # translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        # # Apply the translation (move the resized part)
        # moved_image = cv2.warpAffine(mask, translation_matrix, (image.shape[1], image.shape[0]))

        # # Combine the original image and the translated, resized part
        # # Use bitwise operations to combine the images
        # final_image = cv2.bitwise_or(image, moved_image)

        # Apply perspective transformation (if needed)
        # Define points for perspective transformation
        height, width = image.shape[:2]
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([[rect_x1, rect_y1], [rect_x2, rect_y2],
                        [rect_x3, rect_y3], [rect_x4, rect_y4]])

        # Get the perspective transformation matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply the perspective transformation
        rotated_image = cv2.warpPerspective(image, M, (width, height))
        
        return (cv22tensor(rotated_image),)

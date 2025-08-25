import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

import os
import random
import numpy as np
from typing import Tuple, List, Dict, Optional

# ======================================================================================
# PREVIOUSLY DEFINED MODULES (with slight modifications for training)
# ======================================================================================

# --- Module 1: Face Splitting ---
# (We will use this inside the Dataset class)
import cv2
import mediapipe as mp

def preprocess_and_split_face(
    image: Image.Image, 
    output_size: Tuple[int, int] = (224, 112)
) -> Optional[Tuple[Image.Image, Image.Image]]:
    """Takes a PIL image, detects face, and returns PIL images of left/right halves."""
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR 
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    
    results = face_mesh.process(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    ih, iw, _ = open_cv_image.shape
    landmarks_pixels = np.array([(pt.x * iw, pt.y * ih) for pt in face_landmarks.landmark])

    x_min, y_min = np.min(landmarks_pixels, axis=0).astype(int)
    x_max, y_max = np.max(landmarks_pixels, axis=0).astype(int)
    
    centerline_x = int(np.mean([landmarks_pixels[i, 0] for i in [168, 10, 1, 152]]))
    
    # Use original image (Pillow format) for cropping
    right_face_img = image.crop((x_min, y_min, centerline_x, y_max))
    left_face_img = image.crop((centerline_x, y_min, x_max, y_max))

    if right_face_img.size[0] == 0 or left_face_img.size[0] == 0:
        return None
        
    return left_face_img, right_face_img

# --- Module 2: Siamese Network Model ---
class SiameseNetwork(nn.Module):
    def __init__(self, backbone: str = 'resnet18'):
        super(SiameseNetwork, self).__init__()
        if backbone == 'resnet18':
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.cnn.fc = nn.Identity()
        else:
            raise ValueError("Unsupported backbone")

    def forward_one(self, x):
        return self.cnn(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# ======================================================================================
# NEW MODULES FOR TRAINING
# ======================================================================================

# --- Module 3: Custom Dataset for Generating Pairs ---
class SiamesePairDataset(Dataset):
    def __init__(self, root_dir: str, transform: T.Compose):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_images: Dict[str, List[str]] = self._find_classes()
        self.classes = list(self.class_to_images.keys())

    def _find_classes(self) -> Dict[str, List[str]]:
        class_to_images = {}
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
                # Ensure class has at least 2 images for positive pairing
                if len(images) > 1:
                    class_to_images[class_name] = images
        return class_to_images

    def __len__(self) -> int:
        # Return a large number, we generate pairs on the fly
        return len(self.classes) * 50

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        while True: # Loop until a valid pair is created
            try:
                # Decide if this will be a similar (positive) or dissimilar (negative) pair
                if index % 2 == 0:
                    # Positive pair (label 1)
                    label = 1.0
                    class_name = random.choice(self.classes)
                    img_path1, img_path2 = random.sample(self.class_to_images[class_name], 2)
                else:
                    # Negative pair (label 0)
                    label = 0.0
                    class1, class2 = random.sample(self.classes, 2)
                    img_path1 = random.choice(self.class_to_images[class1])
                    img_path2 = random.choice(self.class_to_images[class2])
                
                # Load the full face image
                full_face_img1 = Image.open(img_path1).convert("RGB")
                
                # Split face and apply transforms
                halves = preprocess_and_split_face(full_face_img1)
                if halves is None: continue # Try again if face detection fails

                left_half, _ = halves # We only need one half per image for this example
                
                # For the second image, we do the same
                full_face_img2 = Image.open(img_path2).convert("RGB")
                halves2 = preprocess_and_split_face(full_face_img2)
                if halves2 is None: continue

                left_half2, _ = halves2

                # Apply the same transformations to both half-faces
                tensor1 = self.transform(left_half)
                tensor2 = self.transform(left_half2)

                return tensor1, tensor2, torch.tensor(label, dtype=torch.float32)

            except Exception as e:
                # print(f"Skipping a bad pair due to error: {e}")
                index +=1 # try next index logic
                continue

# --- Module 4: Contrastive Loss Function ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# --- Module 5: Training Function ---
def train(model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data1, data2, label) in enumerate(train_loader):
        data1, data2, label = data1.to(device), data2.to(device), label.to(device)

        optimizer.zero_grad()
        output1, output2 = model(data1, data2)
        loss = loss_fn(output1, output2, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 20 == 19: # Print every 20 batches
            print(f"Train Epoch: {epoch} [{batch_idx * len(data1)}/{len(train_loader.dataset)}] "
                  f"Loss: {running_loss / 20:.6f}")
            running_loss = 0.0

# --- Module 6: Evaluation Function ---
def evaluate(model, device, test_loader, loss_fn, threshold=1.0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data1, data2, label in test_loader:
            data1, data2, label = data1.to(device), data2.to(device), label.to(device)
            output1, output2 = model(data1, data2)
            
            test_loss += loss_fn(output1, output2, label).item()
            
            # Calculate accuracy based on distance and threshold
            distance = F.pairwise_distance(output1, output2)
            # Prediction: 1 if distance is small (similar), 0 if large (dissimilar)
            pred = (distance < threshold).float()
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# ======================================================================================
# MAIN EXECUTION BLOCK
# ======================================================================================
if __name__ == '__main__':
    # --- Configuration ---
    TRAIN_DIR = 'thermal_data/train'
    TEST_DIR = 'thermal_data/test'
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.0005
    MARGIN = 2.0
    
    # --- Create dummy data directories if they don't exist ---
    # This helps verify the required structure.
    # In practice, you would populate these folders with your actual data.
    print("Verifying data directories...")
    os.makedirs(os.path.join(TRAIN_DIR, 'healthy'), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, 'condition_A'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, 'healthy'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, 'condition_A'), exist_ok=True)
    print("Please ensure your data is in these directories.")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((224, 112)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- DataLoaders ---
    train_dataset = SiamesePairDataset(root_dir=TRAIN_DIR, transform=transform)
    test_dataset = SiamesePairDataset(root_dir=TEST_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Model, Loss, and Optimizer ---
    model = SiameseNetwork(backbone='resnet18').to(device)
    loss_fn = ContrastiveLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training and Evaluation Loop ---
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, loss_fn, epoch)
        evaluate(model, device, test_loader, loss_fn, threshold=MARGIN/2) # A common starting threshold

    # --- Save the trained model ---
    print("Training finished. Saving model...")
    torch.save(model.state_dict(), "siamese_thermal_model.pth")
    print("Model saved to siamese_thermal_model.pth")
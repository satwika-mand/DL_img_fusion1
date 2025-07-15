import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# -------- Device setup --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Debug show image --------
def debug_show_image(tensor, title="Image"):
    img = tensor.squeeze().cpu().detach().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# -------- Autoencoder model --------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# -------- Transform for images --------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# -------- Load grayscale image --------
def load_image(path):
    img = Image.open(path).convert("L")  # grayscale
    return transform(img).unsqueeze(0).to(device)

# -------- Fuse two feature maps --------
def fuse_features(f1, f2):
    return (f1 + f2) / 2

# -------- Save fused output as image --------
def save_fused_image(tensor, path):
    img = tensor.squeeze().cpu().detach().numpy() * 255
    cv2.imwrite(path, img.astype(np.uint8))

# -------- Train the autoencoder --------
def train_autoencoder(ae, img, epochs=1000, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    ae.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        _, output = ae(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return ae

# -------- Main pipeline --------
if __name__ == "__main__":

    # Initialize autoencoder
    ae = Autoencoder().to(device)

    # Check if model is already trained
    if os.path.exists("autoencoder.pth"):
        ae.load_state_dict(torch.load("autoencoder.pth"))
        print("Loaded existing model weights.")
    else:
        print("Starting fresh autoencoder.")

    # Load IR helicopter image
    ir = load_image(
        "tno_dataset/TNO_Image_Fusion_Dataset/Athena_images/helicopter/IR_helib_011.bmp"
    )

    # Train the autoencoder further
    ae = train_autoencoder(ae, ir, epochs=1000, lr=1e-3)

    # Save the trained model
    torch.save(ae.state_dict(), "autoencoder.pth")

    # Visualize decoded IR image
    _, decoded_ir = ae(ir)
    debug_show_image(decoded_ir, title="Decoded IR after retraining")

    # -------- FUSION PROCESS --------

    # Load VIS helicopter image
    vis = load_image(
        "tno_dataset/TNO_Image_Fusion_Dataset/Athena_images/helicopter/VIS_helib_011.bmp"
    )

    # Encode IR and VIS
    feat_ir, _ = ae(ir)
    feat_vis, _ = ae(vis)

    # Fuse the features
    fused_feat = fuse_features(feat_ir, feat_vis)

    # Decode fused features
    fused_img = ae.decoder(fused_feat)

    # Save and show fused helicopter image
    save_fused_image(fused_img, "fused_output.png")
    debug_show_image(fused_img, title="Fused Helicopter Image After Training")

    print("Fusion complete. Fused image saved as 'fused_output.png'.")

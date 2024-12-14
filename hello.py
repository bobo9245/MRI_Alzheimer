import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import numpy as np
import matplotlib.pyplot as plt

# Attention Layer Definition
class Attention3D(nn.Module):
    def __init__(self, in_channels):
        super(Attention3D, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        query = self.query(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, D * H * W)
        value = self.value(x).view(batch_size, -1, D * H * W)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)

        out = self.gamma * out + x
        return out

# Encoder Definition
class Encoder3D(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder3D, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.attention = Attention3D(64)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 8 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.attention(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder Definition
class Decoder3D(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Decoder3D, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8 * 8)
        self.deconv1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attention = Attention3D(32)
        self.deconv2 = nn.ConvTranspose3d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 8, 8, 8)
        x = F.relu(self.deconv1(x))
        x = self.attention(x)
        x = F.softmax(self.deconv2(x), dim=1)
        return x

# VAE Classifier Definition
class VAE3D_Classifier(nn.Module):
    def __init__(self, input_channels, latent_dim, num_classes):
        super(VAE3D_Classifier, self).__init__()
        self.encoder = Encoder3D(input_channels, latent_dim)
        self.decoder = Decoder3D(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        class_probs = self.decoder(z)
        return class_probs, mu, logvar

# Instantiate Model
input_channels = 1
latent_dim = 128
num_classes = 3  # 예: 정상, MCI, 알츠하이머
model = VAE3D_Classifier(input_channels, latent_dim, num_classes)

# 샘플 데이터 생성
x = torch.randn((1, 1, 32, 32, 32))
class_probs, mu, logvar = model(x)

print("Input Shape:", x.shape)
print("Class Probabilities Shape:", class_probs.shape)
print("Mu Shape:", mu.shape)
print("Logvar Shape:", logvar.shape)

# XAI: SHAP 적용
# Wrapper to make the model return just the predictions for SHAP compatibility
def model_wrapper(input_data):
    with torch.no_grad():
        class_probs, _, _ = model(input_data)
        return class_probs

# SHAP explainer setup
def shap_explainer(model, sample_data):
    model.eval()
    explainer = shap.DeepExplainer(model_wrapper, sample_data)
    shap_values = explainer.shap_values(sample_data)
    return shap_values

# 예시 SHAP 값 계산
sample_data = torch.randn((10, 1, 32, 32, 32))
shap_values = shap_explainer(model, sample_data)
print("SHAP Values Computed.")

# 시각화 예시
shap.image_plot([shap_values], sample_data.numpy())

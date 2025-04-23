import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import sys
from network import SimpleCNN
"""
Bu kod CIFAR veri kümesini kullanarak eğitilmiş bir modelden tahmin yapar
"""


def load_classes(data_dir: str=None):
    """
    Sınıf isimlerini data klasöründen okumak için aşağıdaki kodu kullanın.
    
    train_dir = os.path.join(data_dir, 'train')
    classes = sorted(entry.name for entry in os.scandir(train_dir) if entry.is_dir())
    print(classes)
    Sınıfları biliyorsanız aşağıdaki gibi döndürebilirsiniz.
    """
    classes= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return classes



def classify_image(image_path: str,
                   model_path: str = 'simple_cnn_trained.pth',
                   data_dir: str = 'data'):
    # Aygıt
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Sınıf isimleri
    classes = load_classes(data_dir)
    num_classes = len(classes)

    # Model sınıfını oku ve modeli yükle
    model = SimpleCNN(num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # Gelen resimleri boyutlandır, tensöre çevir ve normalize et.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Resmi aç
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)  # batch (yığın) boyutunu ekle

    
    with torch.no_grad():
        logits = model(x)
        probs = nn.functional.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_class = classes[pred_idx.item()]
    confidence = conf.item() * 100.0
    print(f"Tahmin: {pred_class} (Kendime %{confidence:.2f} güveniyorum.)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Kullanımı: python test.py <Resmin konumu> [modelin konumu] [veri klasörü (opsiyonel)]")
        sys.exit(1)

    img_path = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) >= 3 else 'simple_cnn_trained.pth'
    data_directory = sys.argv[3] if len(sys.argv) >= 4 else None
    classify_image(img_path, model_file, data_directory)

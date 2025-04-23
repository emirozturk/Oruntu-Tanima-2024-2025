import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from network import SimpleCNN
"""
Bu kod CIFAR veri kümesini kullanarak bir görüntü sınıflandırma
CNN'i eğitir.
"""

def train(model, device, loader, criterion, optimizer, epoch):
    # Modeli eğitim moduna geçirir
    model.train()
    running_loss = 0.0  # Bir döngü (epoch) boyunca biriken kayıp değeri

    # Veri kümesini batch batch dolaş
    for batch_idx, (data, target) in enumerate(loader):
        # Veriyi ve hedefleri uygun cihaza (GPU/CPU) taşır
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Önceki adımın gradyanlarını sıfırlar
        output = model(data)   # Modelden tahminleri alır
        loss = criterion(output, target)  # Kayıp fonksiyonunu hesaplar
        loss.backward()        # Geriye yayılım yaparak gradyanları hesaplar
        optimizer.step()       # Ağırlıkları günceller

        # Kayıp değerini Python float olarak toplar
        running_loss += loss.detach().item()

        # Her 10 batch'te bir ara çıktı verir
        if batch_idx % 10 == 0:
            print(f"Döngü {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}]  Kayıp: {loss.detach().item():.4f}")

    # Ortalama kaybı hesapla ve ekrana yaz
    avg_loss = running_loss / len(loader)
    print(f"Döngü {epoch} için eğitim Kaybı: {avg_loss:.4f}\n")


def validate(model, device, loader, criterion):
    # Modeli değerlendirme moduna geçirir (Dropout, BatchNorm etkisiz)
    model.eval()
    val_loss = 0.0  # Doğrulama kaybı
    correct = 0     # Doğru sınıflandırma sayısı

    # Gradyan hesaplamayı kapatır (memory ve hız optimizasyonu)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)        # Model çıktısını al
            loss = criterion(output, target)  # Kayıp hesapla
            val_loss += loss.detach().item()  # Kayıp değerini topla

            # Tahmin edilen sınıfı bul ve doğru sayısını güncelle
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    # Ortalama kaybı ve başarı oranını hesapla
    val_loss /= len(loader)
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"Doğrulama (Validasyon) Kaybı: {val_loss:.4f}, Doğruluk: %{accuracy:.2f}\n")

def main():
    # ---------- Yapılandırma ----------
    data_dir = 'data_cifar'  # Eğitim ve doğrulama klasörlerini içeren kök dizin
    batch_size = 32          # Batch (yığın) boyutu
    epochs = 10              # Toplam epoch (döngü) sayısı
    learning_rate = 0.001    # Öğrenme oranı

    # ---------- Cihaz Ayarı (GPU/MPS/CPU) ----------
    device = torch.device('cuda' if torch.cuda.is_available() \
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}\n")

    # ---------- Veri Dönüşümleri ----------
    transform = transforms.Compose([
        # Görüntüleri tensöre çevir
        transforms.ToTensor(),
        # ImageNet istatistikleri ile normalizasyon
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---------- Dataset ve DataLoader Oluşturma ----------
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

     # ---------- Model, Kayıp Fonksiyonu ve Optimizatör ----------
    num_classes = len(train_dataset.classes)
    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Sınıf sayısı: {num_classes}\nSınıflar: {train_dataset.classes}\n")

    # Training and validation loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        validate(model, device, val_loader, criterion)

    # Save the trained model
    torch.save(model.state_dict(), 'simple_cnn.pth')
    print("Eğitim tamamlandı model kaydedildi.")


if __name__ == '__main__':
    main()

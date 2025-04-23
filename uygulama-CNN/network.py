import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Basit bir CNN (Convolutional Neural Network) sınıfı.
    Amaç: CIFAR-10/CIFAR-100 gibi 32x32 renkli görüntüler üzerinde sınıflandırma yapmak.
    
    Mimarisi:
      1) Özellik çıkarımı (feature extraction) katmanları:
         - Conv2d: 3 kanallı (RGB) girdi → 16 kanallı ara katman
         - ReLU: Doğrusal olmayan aktivasyon
         - MaxPool2d: Uzaysal boyutları yarıya indirme
         - Conv2d: 16 → 32 kanal
         - ReLU
         - MaxPool2d
      2) Sınıflandırma (classification) katmanları:
         - Flatten: Çok boyutlu tensoru tek vektöre çevirme
         - Linear: 32×8×8 → 128 birim (çekirdek tam bağlantı)
         - ReLU
         - Linear: 128 → num_classes (çıktı sınıf sayısı)
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes (int): Modelin sınıflandıracağı çıktı sınıfı sayısı.
        """
        super(SimpleCNN, self).__init__()
        
        # --------------------------------------------------------------
        # Özellik çıkarıcı (feature extractor) katman bloğu
        # --------------------------------------------------------------
        self.features = nn.Sequential(
            # 1. Konvolüsyon katmanı: 3 kanaldan 16 kanala geçiş
            # kernel_size=3: 3x3 filtre, padding=1: kenarlarda sıfır doldurarak
            # çıktı boyutunun girdiyle aynı kalmasını sağlar (32x32).
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            # ReLU aktivasyonu: Negatif değerleri 0’a çevirir, doğrusal olmayanlık katar.
            nn.ReLU(inplace=True),
            # Max-pooling: 2x2 bölge içerisinden maksimum değeri seçer,
            # stride=2 ile hem yüksekliği hem genişliği yarıya indirir (→16x16).
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2. Konvolüsyon katmanı: 16 kanaldan 32 kanala geçiş
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # İkinci max-pooling: 16x16 → 8x8
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Not: Girdi görüntüsü 32x32; iki kez 2x2 pool sonrası
        # özellik haritası (feature map) boyutu: 32 kanal × 8 × 8

        # --------------------------------------------------------------
        # Sınıflandırıcı (classifier) katman bloğu
        # --------------------------------------------------------------
        self.classifier = nn.Sequential(
            # Çok boyutlu tensoru (batch_size, 32, 8, 8) → (batch_size, 32*8*8)
            nn.Flatten(),
            # Tam bağlantılı katman: 32*8*8=2048 boyut → 128 boyut
            nn.Linear(in_features=32 * 8 * 8, out_features=128),
            nn.ReLU(inplace=True),
            # Son tam bağlantı: 128 boyut → num_classes boyut (örn. 10)
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri besleme (forward pass) işlemi.
        
        Args:
            x (torch.Tensor): Boyutu (batch_size, 3, 32, 32) olan girdi görüntü tensorü.
        
        Returns:
            torch.Tensor: Boyutu (batch_size, num_classes) olan çıktı skorları (logit’ler).
        """
        # 1) Özellik çıkarımı katmanları üzerinden geçirme
        x = self.features(x)
        # 2) Sınıflandırıcı katmanları üzerinden geçirme
        x = self.classifier(x)
        return x

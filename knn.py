import math

data = {
    "X1": [1, 2, 3, 6, 7, 8],
    "X2": [2, 3, 1, 7, 8, 6],
    "sinif": ["A", "A", "B", "B", "B", "A"]  # Sınıflar
}

def oklid_mesafesi(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def knn_tahmin(x_yeni, y_yeni, k):
    """KNN algoritmasını kullanarak yeni bir noktanın sınıfını tahmin eder."""
    mesafeler = []
    
    # Tüm noktalar için mesafeyi hesapla
    for i in range(len(data["X1"])):
        x = data["X1"][i]
        y = data["X2"][i]
        sinif = data["sinif"][i]
        mesafe = oklid_mesafesi(x, y, x_yeni, y_yeni)
        mesafeler.append((mesafe, sinif))
    
    # Mesafelere göre sıralama
    print(mesafeler)
    mesafeler.sort()
    print(mesafeler)
    
    # En yakın k komşunun sınıflarını al
    sinif_sayaci = {}
    for i in range(k):
        sinif = mesafeler[i][1]
        if sinif in sinif_sayaci:
            sinif_sayaci[sinif] += 1
        else:
            sinif_sayaci[sinif] = 1
    
    # En çok tekrar eden sınıfı bul
    
    tahmin_edilen_sinif = None
    maks_tekrar = 0
    for sinif, tekrar_sayisi in sinif_sayaci.items():
        if tekrar_sayisi > maks_tekrar:
            maks_tekrar = tekrar_sayisi
            tahmin_edilen_sinif = sinif
    """
    Burada örnek sinif_sayaci = {"A": 2, "B": 1}
    Yukarıdaki döngü şöyle kısaltılabilir:
    # tahmin_edilen_sinif = max(sinif_sayaci, key=sinif_sayaci.get)

    """
    
    return tahmin_edilen_sinif

# Yeni bir nokta için tahmin yapalım
yeni_nokta = (5, 5)  # Yeni noktanın X1 ve X2 koordinatları
k_degeri = 3  # Kaç komşuya bakacağımız

tahmin = knn_tahmin(yeni_nokta[0], yeni_nokta[1], k_degeri)
print(f"Yeni nokta {yeni_nokta} için tahmin edilen sınıf: {tahmin}")

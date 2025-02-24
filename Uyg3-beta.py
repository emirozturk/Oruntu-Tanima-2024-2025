import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Farklı Beta öncülleri için olasılık dağılımı
theta = np.linspace(0, 1, 100)

# Adil zar öncülü (hafif öncül)
alpha1, beta1 = 2, 10
plt.plot(theta, beta.pdf(theta, alpha1, beta1), label=f"Beta(2,10)")

# Daha güçlü bir adil zar öncülü
alpha2, beta2 = 10, 50
plt.plot(theta, beta.pdf(theta, alpha2, beta2), label=f"Beta(10,50)")

# Hiç öncül bilgisi yok (uniform prior)
alpha3, beta3 = 1, 1
plt.plot(theta, beta.pdf(theta, alpha3, beta3), label=f"Beta(1,1) (Uniform Prior)")

# Adil olmayan zar
alpha4, beta4 = 50, 10
plt.plot(theta, beta.pdf(theta, alpha4, beta4), label=f"Beta(50,10) (Hileli zar)")

plt.xlabel("θ (6 gelme olasılığı)")
plt.ylabel("Olasılık Yoğunluğu")
plt.title("Farklı Beta Dağılımları")
plt.legend()
plt.show()

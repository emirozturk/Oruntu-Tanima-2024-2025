def bayes_theorem(prior, likelihood, marginal_likelihood):
    """
    Bayes Teoremi'ni uygular.
    
    :param prior: P(H) - Hipotezin öncül olasılığı
    :param likelihood: P(E|H) - Hipotez doğru olduğunda gözlemin olasılığı
    :param marginal_likelihood: P(E) - Gözlemin toplam olasılığı
    :return: P(H|E) - Güncellenmiş olasılık (posterior)
    """
    posterior = (likelihood * prior) / marginal_likelihood
    return posterior

# Örnek kullanım
prior = 0.01  # Bir hastalığa sahip olma olasılığı
likelihood = 0.9  # Testin hasta olan birini doğru tespit etme olasılığı
false_positive_rate = 0.05  # Testin yanlış pozitif verme olasılığı
marginal_likelihood = (likelihood * prior) + (false_positive_rate * (1 - prior))

posterior = bayes_theorem(prior, likelihood, marginal_likelihood)
print(f"Test pozitif çıktığında gerçekten hasta olma olasılığı: {posterior:.4f}")

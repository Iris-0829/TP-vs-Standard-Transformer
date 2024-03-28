import matplotlib.pyplot as plt

models = ['BERT', 'GPT2', 'TP']
perplexities = [7593, 8103, 10858]
losses = [8.93, 9, 9.29]

# Perplexity
plt.figure(figsize=(8, 6))
plt.bar(models, perplexities)
plt.xlabel('Model')
plt.ylabel('Perplexity')
plt.title('Perplexity Comparison')
plt.show()

# Loss
plt.figure(figsize=(8, 6))
plt.bar(models, losses)
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.show()

# Strided validation comparison
strided_loss = [8.48, 8.55, 8.82]
strided_perplexities = [4862, 5177, 6789]

# Perplexity
plt.figure(figsize=(8, 6))
plt.bar(models, strided_perplexities)
plt.xlabel('Model')
plt.ylabel('Perplexity (strided)')
plt.title('Perplexity (strided) Comparison')
plt.show()

# Loss
plt.figure(figsize=(8, 6))
plt.bar(models, strided_loss)
plt.xlabel('Model')
plt.ylabel('Loss (strided)')
plt.title('Loss (strided) Comparison')
plt.show()
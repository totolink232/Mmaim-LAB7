import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Створення директорії для графіків
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Встановлюємо фіксований seed для відтворюваності результатів
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("ЛАБОРАТОРНА РОБОТА № 7. ПРИКЛАД ЗІ СЛАЙДІВ (PyTorch SimpleNN)")
print("=" * 70)

# 1. Генерація тестового набору даних у 2D-просторі на дві категорії
# Створюємо 2 хмари точок (бінарна класифікація)
n_samples = 200
X_class0 = np.random.randn(n_samples // 2, 2) * 0.7 + np.array([-1.0, -1.0])
X_class1 = np.random.randn(n_samples // 2, 2) * 0.7 + np.array([1.2, 1.2])

X_np = np.vstack([X_class0, X_class1]).astype(np.float32)
y_np = np.vstack([np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))]).astype(np.float32)

# Конвертуємо дані в тензори PyTorch
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

# 2. Архітектура моделі SimpleNN згідно зі слайдом
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)       # вхідний шар (2 входи) -> прихований (4 нейрони)
        self.fc2 = nn.Linear(4, 1)       # прихований шар -> вихідний шар (1 нейрон)
        self.activation = nn.Sigmoid()   # функція активації (сигмоїда)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x

# 3. Ініціалізація моделі, критерію втрат та оптимізатора (зі слайда)
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("\nАрхітектура моделі SimpleNN:")
print(model)

# 4. Навчання моделі
epochs = 400
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Пряме поширення (forward pass)
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Зворотне поширення помилки (backpropagation)
    loss.backward()
    optimizer.step()
    
    # Розрахунок точності (accuracy)
    preds = (outputs >= 0.5).float()
    acc = (preds == y).float().mean().item()
    
    loss_history.append(loss.item())
    accuracy_history.append(acc)
    
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Епоха [{epoch+1:3d}/{epochs}] | Втрати (MSE): {loss.item():.5f} | Точність: {acc * 100:.2f}%")

# 5. Візуалізація результатів класифікації та розділової границі
plt.figure(figsize=(12, 5))

# Графік 1: Розділова поверхня та класи точок
plt.subplot(1, 2, 1)
x_min, x_max = X_np[:, 0].min() - 0.8, X_np[:, 0].max() + 0.8
y_min, y_max = X_np[:, 1].min() - 0.8, X_np[:, 1].max() + 0.8
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))

model.eval()
with torch.no_grad():
    Z = model(grid_tensor).reshape(xx.shape).numpy()

contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.7)
plt.colorbar(contour, label='Ймовірність належності до класу 1')
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

plt.scatter(X_np[y_np.ravel() == 0, 0], X_np[y_np.ravel() == 0, 1], 
            color='blue', edgecolors='k', s=45, label='Клас 0', alpha=0.9)
plt.scatter(X_np[y_np.ravel() == 1, 0], X_np[y_np.ravel() == 1, 1], 
            color='red', edgecolors='k', s=45, label='Клас 1', alpha=0.9)

plt.title('Розділова межа моделі SimpleNN (2D-простір)', fontsize=11, fontweight='bold')
plt.xlabel('Ознака X1')
plt.ylabel('Ознака X2')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.5)

# Графік 2: Криві зміни помилки та точності під час навчання
plt.subplot(1, 2, 2)
ax1 = plt.gca()
color1 = 'crimson'
ax1.set_xlabel('Епоха навчання')
ax1.set_ylabel('Функція втрат (MSE Loss)', color=color1)
line1 = ax1.plot(loss_history, color=color1, linewidth=2, label='MSE Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle=':', alpha=0.5)

ax2 = ax1.twinx()
color2 = 'teal'
ax2.set_ylabel('Точність (Accuracy)', color=color2)
line2 = ax2.plot(accuracy_history, color=color2, linewidth=2, linestyle='--', label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(-0.05, 1.05)

# Спільна легенда
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')
plt.title('Динаміка помилки та точності SimpleNN', fontsize=11, fontweight='bold')

plt.tight_layout()
example_plot_path = os.path.join(PLOTS_DIR, 'task1_example_classification.png')
plt.savefig(example_plot_path, dpi=300)
plt.close()

print(f"\nГрафік базового прикладу збережено у: {example_plot_path}")
print(f"Фінальна точність моделі: {accuracy_history[-1] * 100:.2f}%, фінальна помилка: {loss_history[-1]:.5f}")

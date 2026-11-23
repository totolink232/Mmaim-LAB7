import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import os

# Створення директорії для графіків
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Фіксація генератора випадкових чисел для повної відтворюваності
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("ЛАБОРАТОРНА РОБОТА № 7. ДОСЛІДЖЕННЯ ШТУЧНИХ НЕЙРОННИХ МЕРЕЖ")
print("Тема: Побудова та аналіз параметрів ШНМ для класифікації у 2D-просторі")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. ГЕНЕРАЦІЯ ТА ПІДГОТОВКА НЕЛІНІЙНОГО ДАТАСЕТУ
# -----------------------------------------------------------------------------
# Створюємо нелінійну вибірку "два місяці" (make_moons) - 300 зразків із шумом
n_samples = 300
noise_level = 0.20
X_raw, y_raw = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)

# Розділяємо на навчальну (75%) та тестову (25%) вибірки
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_raw, y_raw, test_size=0.25, random_state=42, stratify=y_raw
)

# Перетворення в тензори PyTorch
X_train = torch.from_numpy(X_train_np.astype(np.float32))
y_train = torch.from_numpy(y_train_np.astype(np.float32)).unsqueeze(1)
X_test = torch.from_numpy(X_test_np.astype(np.float32))
y_test = torch.from_numpy(y_test_np.astype(np.float32)).unsqueeze(1)

print(f"Розмірність навчальної вибірки: {X_train.shape[0]} точок")
print(f"Розмірність тестової вибірки:    {X_test.shape[0]} точок")

# Візуалізація вихідного датасету
plt.figure(figsize=(8, 6))
plt.scatter(X_train_np[y_train_np == 0, 0], X_train_np[y_train_np == 0, 1], 
            color='royalblue', marker='o', s=50, edgecolors='k', label='Навчальна: Клас 0', alpha=0.85)
plt.scatter(X_train_np[y_train_np == 1, 0], X_train_np[y_train_np == 1, 1], 
            color='crimson', marker='o', s=50, edgecolors='k', label='Навчальна: Клас 1', alpha=0.85)
plt.scatter(X_test_np[y_test_np == 0, 0], X_test_np[y_test_np == 0, 1], 
            color='deepskyblue', marker='^', s=70, edgecolors='k', label='Тестова: Клас 0', alpha=0.95)
plt.scatter(X_test_np[y_test_np == 1, 0], X_test_np[y_test_np == 1, 1], 
            color='darkorange', marker='^', s=70, edgecolors='k', label='Тестова: Клас 1', alpha=0.95)

plt.title('2D-датасет нелінійної класифікації (make_moons)', fontsize=12, fontweight='bold')
plt.xlabel('Координата X1', fontsize=11)
plt.ylabel('Координата X2', fontsize=11)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
ds_plot_path = os.path.join(PLOTS_DIR, 'exp_dataset_visualization.png')
plt.savefig(ds_plot_path, dpi=300)
plt.close()
print(f"Графік датасету збережено: {ds_plot_path}")

# -----------------------------------------------------------------------------
# 2. ОПИС МОДУЛЬНОГО КЛАСУ НЕЙРОННОЇ МЕРЕЖІ
# -----------------------------------------------------------------------------
class FlexibleMLP(nn.Module):
    def __init__(self, hidden_dim=8, activation='relu'):
        super(FlexibleMLP, self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        if activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif activation.lower() == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation.lower() == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Невідома функція активації: {activation}")
            
        self.out_act = nn.Sigmoid() # вихідна сигмоїда для бінарної класифікації

    def forward(self, x):
        h = self.act(self.fc1(x))
        out = self.out_act(self.fc2(h))
        return out

# Функція для навчання моделі з логуванням метрик
def train_model(model, X_tr, y_tr, X_te, y_te, epochs=300, lr=0.1, loss_name='bce'):
    if loss_name.lower() == 'bce':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        preds_train = model(X_tr)
        loss_tr = criterion(preds_train, y_tr)
        loss_tr.backward()
        optimizer.step()
        
        # Оцінка метрик
        model.eval()
        with torch.no_grad():
            preds_test = model(X_te)
            loss_te = criterion(preds_test, y_te)
            
            acc_tr = ((preds_train >= 0.5).float() == y_tr).float().mean().item()
            acc_te = ((preds_test >= 0.5).float() == y_te).float().mean().item()
            
        train_losses.append(loss_tr.item())
        test_losses.append(loss_te.item())
        train_accs.append(acc_tr)
        test_accs.append(acc_te)
        
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1],
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    }

# -----------------------------------------------------------------------------
# 3. ОСНОВНЕ НАВЧАННЯ БАЗОВОЇ МОДЕЛІ (16 нейронів, ReLU, lr=0.1)
# -----------------------------------------------------------------------------
base_hidden = 16
base_epochs = 400
base_lr = 0.1

torch.manual_seed(42)
base_model = FlexibleMLP(hidden_dim=base_hidden, activation='relu')
history_base = train_model(base_model, X_train, y_train, X_test, y_test, 
                           epochs=base_epochs, lr=base_lr, loss_name='bce')

print("\nРезультати базової моделі (16 нейронів, ReLU, lr=0.1):")
print(f"  Навчальна точність (Train Acc): {history_base['final_train_acc'] * 100:.2f}%")
print(f"  Тестова точність (Test Acc):     {history_base['final_test_acc'] * 100:.2f}%")
print(f"  Кінцеві втрати на тесті (Loss):  {history_base['final_test_loss']:.5f}")

# Графік 1: Динаміка навчання базової моделі (Loss & Accuracy)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_base['train_losses'], label='Train Loss (BCE)', color='crimson', lw=2)
plt.plot(history_base['test_losses'], label='Test Loss (BCE)', color='navy', lw=2, linestyle='--')
plt.title('Криві функції втрат (BCE Loss) за епохами', fontsize=11, fontweight='bold')
plt.xlabel('Епоха навчання', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(np.array(history_base['train_accs']) * 100, label='Train Accuracy', color='darkgreen', lw=2)
plt.plot(np.array(history_base['test_accs']) * 100, label='Test Accuracy', color='purple', lw=2, linestyle='--')
plt.title('Криві точності (Accuracy %) за епохами', fontsize=11, fontweight='bold')
plt.xlabel('Епоха навчання', fontsize=10)
plt.ylabel('Точність, %', fontsize=10)
plt.ylim(40, 103)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

plt.tight_layout()
train_curve_path = os.path.join(PLOTS_DIR, 'exp_training_loss_accuracy.png')
plt.savefig(train_curve_path, dpi=300)
plt.close()
print(f"Графік кривих навчання збережено: {train_curve_path}")

# -----------------------------------------------------------------------------
# 4. ЕКСПЕРИМЕНТ 1: ВПЛИВ КІЛЬКОСТІ НЕЙРОНІВ У ПРИХОВАНОМУ ШАРІ
# -----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("Експеримент 1: Дослідження кількості нейронів у прихованому шарі")
print("-" * 70)

neurons_list = [2, 4, 8, 16, 32]
models_neurons = {}
results_neurons = []

x_min, x_max = X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5
y_min, y_max = X_raw[:, 1].min() - 0.5, X_raw[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))

fig, axes = plt.subplots(1, len(neurons_list), figsize=(20, 4.2), sharey=True)

for idx, n_neu in enumerate(neurons_list):
    torch.manual_seed(42)
    m = FlexibleMLP(hidden_dim=n_neu, activation='relu')
    hist = train_model(m, X_train, y_train, X_test, y_test, epochs=350, lr=0.1)
    models_neurons[n_neu] = m
    
    results_neurons.append({
        'neurons': n_neu,
        'train_acc': hist['final_train_acc'],
        'test_acc': hist['final_test_acc'],
        'train_loss': hist['final_train_loss'],
        'test_loss': hist['final_test_loss']
    })
    
    print(f"Нейронів: {n_neu:2d} | Train Acc: {hist['final_train_acc']*100:.2f}% | Test Acc: {hist['test_accs'][-1]*100:.2f}% | Test Loss: {hist['final_test_loss']:.4f}")
    
    m.eval()
    with torch.no_grad():
        Z = m(grid_tensor).reshape(xx.shape).numpy()
        
    ax = axes[idx]
    ax.contourf(xx, yy, Z, levels=15, cmap='RdBu_r', alpha=0.65)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    ax.scatter(X_raw[y_raw == 0, 0], X_raw[y_raw == 0, 1], c='royalblue', s=20, edgecolors='k', alpha=0.7)
    ax.scatter(X_raw[y_raw == 1, 0], X_raw[y_raw == 1, 1], c='crimson', s=20, edgecolors='k', alpha=0.7)
    
    ax.set_title(f"{n_neu} нейронів\nTest Acc: {hist['final_test_acc']*100:.1f}%", fontsize=11, fontweight='bold')
    ax.set_xlabel('X1')
    if idx == 0:
        ax.set_ylabel('X2')
    ax.grid(True, linestyle=':', alpha=0.4)

plt.suptitle('Розділові поверхні залежно від кількості нейронів у прихованому шарі', 
             fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
neurons_plot_path = os.path.join(PLOTS_DIR, 'exp_decision_boundaries_neurons.png')
plt.savefig(neurons_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Графік розділових меж збережено: {neurons_plot_path}")

# -----------------------------------------------------------------------------
# 5. ЕКСПЕРИМЕНТ 2: ВПЛИВ ШВИДКОСТІ НАВЧАННЯ (LEARNING RATE)
# -----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("Експеримент 2: Дослідження швидкості навчання (Learning Rate)")
print("-" * 70)

lr_list = [0.001, 0.01, 0.1, 1.0]
results_lr = {}

plt.figure(figsize=(12, 5))
colors_lr = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c']

for idx, lr_val in enumerate(lr_list):
    torch.manual_seed(42)
    m = FlexibleMLP(hidden_dim=16, activation='relu')
    hist = train_model(m, X_train, y_train, X_test, y_test, epochs=300, lr=lr_val)
    results_lr[lr_val] = hist
    
    print(f"LR = {lr_val:5.3f} | Train Acc: {hist['final_train_acc']*100:.2f}% | Test Acc: {hist['final_test_acc']*100:.2f}% | Test Loss: {hist['final_test_loss']:.4f}")
    
    plt.subplot(1, 2, 1)
    plt.plot(hist['train_losses'], label=f'LR = {lr_val}', color=colors_lr[idx], lw=2)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.array(hist['test_accs']) * 100, label=f'LR = {lr_val}', color=colors_lr[idx], lw=2)

plt.subplot(1, 2, 1)
plt.title('Втрати (Train Loss) при різних Learning Rate', fontsize=11, fontweight='bold')
plt.xlabel('Епоха')
plt.ylabel('BCE Loss')
plt.yscale('log')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.title('Тестова точність (Test Accuracy) при різних Learning Rate', fontsize=11, fontweight='bold')
plt.xlabel('Епоха')
plt.ylabel('Точність, %')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

plt.tight_layout()
lr_plot_path = os.path.join(PLOTS_DIR, 'exp_learning_rate_dynamics.png')
plt.savefig(lr_plot_path, dpi=300)
plt.close()
print(f"Графік динаміки швидкості навчання збережено: {lr_plot_path}")

# -----------------------------------------------------------------------------
# 6. ЕКСПЕРИМЕНТ 3: ПОРІВНЯННЯ ФУНКЦІЙ АКТИВАЦІЇ (Sigmoid vs Tanh vs ReLU)
# -----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("Експеримент 3: Порівняння функцій активації (Sigmoid, Tanh, ReLU)")
print("-" * 70)

activations = ['sigmoid', 'tanh', 'relu']
results_act = {}
act_colors = {'sigmoid': '#e67e22', 'tanh': '#2980b9', 'relu': '#27ae60'}

plt.figure(figsize=(12, 5))

for act_name in activations:
    torch.manual_seed(42)
    m = FlexibleMLP(hidden_dim=16, activation=act_name)
    hist = train_model(m, X_train, y_train, X_test, y_test, epochs=300, lr=0.1)
    results_act[act_name] = hist
    
    print(f"Активація: {act_name.upper():7s} | Train Acc: {hist['final_train_acc']*100:.2f}% | Test Acc: {hist['final_test_acc']*100:.2f}% | Test Loss: {hist['final_test_loss']:.4f}")
    
    plt.subplot(1, 2, 1)
    plt.plot(hist['test_losses'], label=f'{act_name.upper()}', color=act_colors[act_name], lw=2)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.array(hist['test_accs']) * 100, label=f'{act_name.upper()}', color=act_colors[act_name], lw=2)

plt.subplot(1, 2, 1)
plt.title('Тестові втрати (Test Loss) для різних функцій активації', fontsize=11, fontweight='bold')
plt.xlabel('Епоха')
plt.ylabel('BCE Loss')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.title('Тестова точність (Test Accuracy) для різних функцій активації', fontsize=11, fontweight='bold')
plt.xlabel('Епоха')
plt.ylabel('Точність, %')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

plt.tight_layout()
act_plot_path = os.path.join(PLOTS_DIR, 'exp_activation_functions_comparison.png')
plt.savefig(act_plot_path, dpi=300)
plt.close()
print(f"Графік порівняння функцій активації збережено: {act_plot_path}")

print("\n" + "=" * 80)
print("Усі практичні завдання та експерименти успішно виконано!")
print("=" * 80)

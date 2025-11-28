import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time

# --- KONFIGURACJA ---
HIDDEN_LAYERS_SIZES = [128]
LEARNING_RATE = 0.2
BATCH_SIZE = 8
EPOCHS = 20

# --- KONFIGURACJA HIPOKAMPU (WALKA) ---
MEMORY_SLOTS = 128        # Liczba prototypów
STRUGGLE_THRESHOLD = 0.1 # Jak duży musi być błąd, żeby hipokamp zareagował?
MAX_STRUGGLE_LOOPS = 4  # Ile razy maksymalnie próbujemy zrozumieć trudny przypadek
ATTENTION_FORCE = 0.002    # Siła zmian wag podczas pętli walki

# --- FUNKCJE MATEMATYCZNE ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# --- HIPOKAMP: INSTANT STRUGGLE ---
class HippocampusInstant:
    def __init__(self, embedding_dim, num_slots=64):
        self.embedding_dim = embedding_dim
        self.num_slots = num_slots
        # Pamięć prototypów (Codebook)
        self.prototypes = np.random.uniform(-0.1, 0.1, (num_slots, embedding_dim))

    def process_struggle(self, layer_obj, single_input, single_current_output):
        """
        Ta funkcja jest wywoływana TYLKO dla trudnego przypadku.
        Wykonuje pętlę dopasowywania wag, aż wyjście zbliży się do prototypu.
        """
        # single_input: (1, input_dim) - wejście do warstwy dla tego 1 przypadku
        # single_current_output: (1, output_dim) - obecne, błędne wyjście

        # 1. Znajdź najbliższy prototyp (do czego to jest podobne?)
        # (x-y)^2
        dists = np.sum((self.prototypes - single_current_output)**2, axis=1)
        best_idx = np.argmin(dists)
        target_prototype = self.prototypes[best_idx]

        # 2. Pętla Walki (Iteracyjne poprawianie wag)
        for i in range(MAX_STRUGGLE_LOOPS):
            # Ponowny forward na TYM SAMYM wejściu z aktualnymi wagami
            # (Musimy przeliczyć, bo wagi się zmieniają w pętli)
            current_z = np.dot(single_input, layer_obj.weights) + layer_obj.bias
            current_a = relu(current_z) # Zakładamy ReLU w warstwie ukrytej

            # Oblicz różnicę strukturalną (Prototyp - Aktualne)
            diff = target_prototype - current_a

            # Jeśli jesteśmy już bardzo blisko, przerywamy walkę (sukces)
            if np.mean(np.abs(diff)) < 0.05:
                # Opcjonalnie: Aktualizujemy prototyp (uczymy się nowego wariantu)
                self.prototypes[best_idx] += diff[0] * 0.01
                return i + 1 # Zwracamy liczbę pętli, które były potrzebne

            # Reguła Delty dla bezpośredniej modyfikacji wag
            # DeltaW = Input.T * Diff * Force
            w_change = np.outer(single_input, diff) * ATTENTION_FORCE
            b_change = diff[0] * ATTENTION_FORCE

            # Aplikacja zmian
            layer_obj.weights += w_change
            layer_obj.bias += b_change

        return MAX_STRUGGLE_LOOPS # Dotarliśmy do limitu

# --- WARSTWA ---
class InjectionLayer:
    def __init__(self, input_size, output_size, activation="relu", name="Layer"):
        self.name = name
        self.activation = activation

        limit = np.sqrt(6 / input_size)
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size)
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        z = np.dot(x, self.weights) + self.bias
        if self.activation == "sigmoid": return sigmoid(z)
        elif self.activation == "softmax": return softmax(z)
        elif self.activation == "relu": return relu(z)
        return z

    def update_weights(self, signal, lr):
        weight_grad = np.dot(self.last_input.T, signal)
        bias_grad = np.sum(signal, axis=0)
        batch_scale = 1.0 / self.last_input.shape[0]

        self.weights += lr * weight_grad * batch_scale
        self.bias += lr * bias_grad * batch_scale

# --- SIEĆ ---
class MnistLearner:
    def __init__(self, hidden_sizes, hip_slots):
        self.layers = []
        input_size = 784

        for i, size in enumerate(hidden_sizes):
            self.layers.append(
                InjectionLayer(input_size, size, activation="relu", name=f"Hidden_{i+1}")
            )
            input_size = size

        # Hipokamp (Instant)
        self.hippocampus = HippocampusInstant(embedding_dim=hidden_sizes[-1], num_slots=hip_slots)

        self.layers.append(
            InjectionLayer(input_size, 10, activation="softmax", name="Output")
        )

    def forward(self, x):
        activations = [x]
        curr = x
        for layer in self.layers:
            curr = layer.forward(curr)
            activations.append(curr)
        return activations

    def train_step(self, x_batch, target_encoded, lr):
        # 1. Standardowy Forward Pass (cały batch)
        activations = self.forward(x_batch)
        final_out = activations[-1]

        # 2. Obliczanie błędu (Loss) dla każdego przykładu z osobna
        # Używamy MSE jako miary błędu dla prostoty
        # error_per_sample: wektor (batch_size,)
        errors = target_encoded - final_out
        error_magnitude = np.mean(np.abs(errors), axis=1)

        # 3. Standardowy Backpropagation (Intencja Sieci) - "Ogólna Nauka"
        # Robimy to zawsze, żeby sieć uczyła się też na łatwych przykładach
        signals = []
        signal_output = errors
        signals.append(signal_output)

        curr_signal = signal_output
        for i in range(len(self.layers) - 2, -1, -1):
            next_layer = self.layers[i+1]
            curr_layer_obj = self.layers[i]
            curr_layer_output = activations[i+1]
            error = np.dot(curr_signal, next_layer.weights.T)

            if curr_layer_obj.activation == "relu": deriv = relu_derivative(curr_layer_output)
            else: deriv = 1.0

            hidden_signal = error * deriv
            signals.append(hidden_signal)
            curr_signal = hidden_signal
        signals.reverse()
        for layer, signal in zip(self.layers, signals):
            layer.update_weights(signal, lr)

        # 4. --- INTERWENCJA HIPOKAMPU: SPRAWDZANIE TRUDNYCH PRZYPADKÓW ---
        # Sprawdzamy, czy któryś przykład w batchu przekroczył próg frustracji

        # Dane z warstwy ukrytej (przed wyjściem), którą zarządza hipokamp
        hidden_layer_obj = self.layers[-2]
        hidden_inputs = activations[-3] # Wejście do warstwy ukrytej
        hidden_outputs = activations[-2] # Wyjście z warstwy ukrytej

        struggle_count = 0

        for i in range(x_batch.shape[0]):
            if error_magnitude[i] > STRUGGLE_THRESHOLD:
                # Znaleziono trudny przypadek!
                struggle_count += 1

                # Izolujemy dane dla tego jednego przypadku
                # Reshape (1, N) jest konieczny, żeby zachować wymiary macierzy
                single_input = hidden_inputs[i].reshape(1, -1)
                single_output = hidden_outputs[i].reshape(1, -1)

                # Zlecamy hipokampowi "walkę" z tym przypadkiem
                # To zmieni wagi hidden_layer_obj w miejscu!
                loops = self.hippocampus.process_struggle(hidden_layer_obj, single_input, single_output)

                # (Opcjonalnie) Możemy wypisać log, jeśli walka była długa
                # if loops > 5:
                #    print(f"   -> Struggle detected! Fixed in {loops} loops. Error was: {error_magnitude[i]:.3f}")

        return final_out, struggle_count

# --- PRZYGOTOWANIE DANYCH ---
print("--- START (INSTANT STRUGGLE REPLAY) ---")
t_start = time.time()

mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int')
X = X.to_numpy()

enc = OneHotEncoder(sparse_output=False)
y_encoded = enc.fit_transform(y.to_numpy().reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

print(f"Dane gotowe w {time.time()-t_start:.2f}s")
print(f"Architektura: Struggle Threshold: {STRUGGLE_THRESHOLD}")
print("-" * 60)

# --- PĘTLA TRENINGOWA ---
net = MnistLearner(HIDDEN_LAYERS_SIZES, hip_slots=MEMORY_SLOTS)

total_struggles = 0

for epoch in range(EPOCHS):
    epoch_start = time.time()
    epoch_struggles = 0

    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for i in range(0, X_train.shape[0], BATCH_SIZE):
        x_batch = X_shuffled[i:i+BATCH_SIZE]
        y_batch = y_shuffled[i:i+BATCH_SIZE]

        # Trening zwraca teraz informację, ile razy musiał walczyć
        _, struggles = net.train_step(x_batch, y_batch, LEARNING_RATE)
        epoch_struggles += struggles

    # Walidacja
    activations = net.forward(X_test)
    out_test = activations[-1]

    preds = np.argmax(out_test, axis=1)
    true_vals = np.argmax(y_test, axis=1)
    acc = np.mean(preds == true_vals) * 100

    print(f"Epoka {epoch+1}/{EPOCHS} | Czas: {time.time()-epoch_start:.2f}s | Dokładność: {acc:.2f}% | Interwencje (Walka): {epoch_struggles}")

print("-" * 60)
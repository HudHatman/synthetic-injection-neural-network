import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time

# --- KONFIGURACJA ---
HIDDEN_LAYERS_SIZES = [256]
LEARNING_RATE = 0.05
BATCH_SIZE = 8
EPOCHS = 3

# --- KONFIGURACJA CYKLU DOBOWEGO MÓZGU ---
MEMORY_SLOTS = 650        # Ile prototypów pamięta hipokamp (pamięć semantyczna)
EPISODIC_BUFFER_SIZE = 50 # Ile "zdarzeń" zapamiętujemy w ciągu dnia (pamięć epizodyczna)
SLEEP_INTERVAL = 10000     # Co ile batchy następuje faza snu (cykl "dzień")
SLEEP_CYCLES = 5         # Ile razy powtarzamy konsolidację podczas jednej nocy

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

# --- HIPOKAMP: FAZY AKTYWNOŚCI ---
class HippocampusPhasic:
    def __init__(self, embedding_dim, num_slots=64, buffer_size=200):
        self.embedding_dim = embedding_dim
        self.num_slots = num_slots
        self.buffer_size = buffer_size

        # 1. Pamięć Semantyczna (Długotrwała - Prototypy)
        self.prototypes = np.random.uniform(-0.1, 0.1, (num_slots, embedding_dim))

        # 2. Pamięć Epizodyczna (Krótkotrwała - Bufor dnia)
        # Przechowuje krotki: (input_state, hidden_state) z trudnych przypadków
        self.episodic_buffer = []

        self.phase = "WAKE" # WAKE lub SLEEP

    def wake_process(self, layer_input, layer_output, error_magnitude):
        """
        FAZA CZUWANIA:
        Hipokamp tylko obserwuje. Jeśli napotka coś 'trudnego' (duży błąd),
        zapisuje to w pamięci epizodycznej do późniejszego przemyślenia.
        """
        # Jeśli bufor pełny, usuwamy najstarsze wspomnienie (zapominanie)
        if len(self.episodic_buffer) >= self.buffer_size:
            self.episodic_buffer.pop(0)

        # Zapisujemy zdarzenie.
        # Ważne: Zapisujemy kopię danych (snapshot), a nie referencję!
        # Decyzja o zapisie może zależeć od błędu (tu: zapisujemy losowo lub wszystko dla uproszczenia)
        # W biologii: emocje/nowość decydują o zapisie.
        if np.mean(np.abs(error_magnitude)) > 0.1: # Próg "ciekawości"
            self.episodic_buffer.append((layer_input.copy(), layer_output.copy()))

        # Szybka aktualizacja prototypów (Fast Learning) - hipokamp uczy się natychmiast
        # Znajdź najbliższy prototyp i lekko go przesuń
        # (To symuluje plastyczność synaptyczną samego hipokampu, nie kory)
        inputs_sq = np.sum(layer_output**2, axis=1, keepdims=True)
        proto_sq = np.sum(self.prototypes**2, axis=1)
        dists = inputs_sq + proto_sq - 2 * np.dot(layer_output, self.prototypes.T)
        closest = np.argmin(dists, axis=1)

        alpha = 0.1 # Szybkie uczenie
        for i, idx in enumerate(closest):
            self.prototypes[idx] = self.prototypes[idx] * (1-alpha) + layer_output[i] * alpha

    def sleep_process(self, layer_obj, cycles=5):
        """
        FAZA SNU (Konsolidacja):
        Odtwarzamy zdarzenia z bufora epizodycznego.
        Wymuszamy na korze (warstwie), by jej wagi dopasowały się do prototypów.
        """
        if not self.episodic_buffer:
            return

        print(f"   [HIPPOCAMPUS] Entering SLEEP Phase... Consolidating {len(self.episodic_buffer)} episodes.")

        # Konwersja bufora na batch do obliczeń
        # Zbieramy wszystkie zapamiętane wejścia i wyjścia
        # inputs_mem: co weszło do warstwy, outputs_mem: co wyszło (wtedy)
        inputs_mem = np.array([e[0] for e in self.episodic_buffer])
        # Spłaszczamy listę batchy w jeden duży tensor
        inputs_mem = np.vstack(inputs_mem)

        outputs_mem = np.array([e[1] for e in self.episodic_buffer])
        outputs_mem = np.vstack(outputs_mem)

        # Iteracje konsolidacji (Sharp-Wave Ripples)
        for _ in range(cycles):
            # 1. Dopasowanie wspomnień do prototypów (gdzie to powinno należeć?)
            inputs_sq = np.sum(outputs_mem**2, axis=1, keepdims=True)
            proto_sq = np.sum(self.prototypes**2, axis=1)
            dists = inputs_sq + proto_sq - 2 * np.dot(outputs_mem, self.prototypes.T)

            closest_indices = np.argmin(dists, axis=1)
            target_prototypes = self.prototypes[closest_indices]

            # 2. Obliczenie korekty wag (Structural Adjustment)
            # Przesuwamy wagi tak, by dla danego inputu dawały wynik bliższy prototypowi
            structural_error = (target_prototypes - outputs_mem)

            # Wzór Delty: DeltaW = Input.T * Error * Force
            # Force jest duży, bo sen ma potężny wpływ na plastyczność
            sleep_force = 0.05
            batch_scale = 1.0 / inputs_mem.shape[0]

            w_correction = np.dot(inputs_mem.T, structural_error) * sleep_force * batch_scale
            b_correction = np.sum(structural_error, axis=0) * sleep_force * batch_scale

            # Aplikacja zmian bezpośrednio do warstwy
            layer_obj.weights += w_correction
            layer_obj.bias += b_correction

        # Po nocy czyścimy bufor epizodyczny (gotowość na nowy dzień)
        self.episodic_buffer = []

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
        """Standardowy Backprop (tylko w fazie Wake)"""
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

        # Warstwy
        for i, size in enumerate(hidden_sizes):
            self.layers.append(
                InjectionLayer(input_size, size, activation="relu", name=f"Hidden_{i+1}")
            )
            input_size = size

        # Hipokamp (fazowy)
        self.hippocampus = HippocampusPhasic(embedding_dim=hidden_sizes[-1],
                                             num_slots=hip_slots,
                                             buffer_size=EPISODIC_BUFFER_SIZE)

        # Output
        self.layers.append(
            InjectionLayer(input_size, 10, activation="softmax", name="Output")
        )

        self.batch_counter = 0

    def forward(self, x):
        activations = [x]
        curr = x
        for layer in self.layers:
            curr = layer.forward(curr)
            activations.append(curr)
        return activations

    def train_step_wake(self, x, target_encoded, lr):
        """FAZA CZUWANIA: Normalny trening + zbieranie wspomnień"""
        self.batch_counter += 1

        # 1. Forward
        activations = self.forward(x)
        final_out = activations[-1]

        # Dane dla hipokampu (z ostatniej warstwy ukrytej)
        hidden_input = activations[-3]
        hidden_output = activations[-2]

        # 2. Backpropagation (Standard)
        signals = []
        signal_output = (target_encoded - final_out) # To jest też miara błędu/zaskoczenia
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

        # 3. INTERWENCJA HIPOKAMPU (ZAPIS)
        # Przekazujemy sygnał błędu, żeby hipokamp wiedział, czy warto to zapamiętać
        self.hippocampus.wake_process(hidden_input, hidden_output, signal_output)

        # 4. Sprawdzenie czy czas na sen
        if self.batch_counter % SLEEP_INTERVAL == 0:
            self.sleep_routine()

        return final_out

    def sleep_routine(self):
        """Uruchamia proces konsolidacji"""
        # Hipokamp przejmuje kontrolę nad warstwą ukrytą
        # Modyfikuje wagi warstwy 'Hidden_X' (przedostatnia w liście layers)
        target_layer = self.layers[-2]
        self.hippocampus.sleep_process(target_layer, cycles=SLEEP_CYCLES)

# --- PRZYGOTOWANIE DANYCH ---
print("--- START (PHASIC BRAIN: WAKE & SLEEP) ---")
t_start = time.time()

mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int')
X = X.to_numpy()

enc = OneHotEncoder(sparse_output=False)
y_encoded = enc.fit_transform(y.to_numpy().reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

print(f"Dane gotowe w {time.time()-t_start:.2f}s")
print(f"Architektura: Cykl dobowy co {SLEEP_INTERVAL} batchy.")
print("-" * 60)

# --- PĘTLA TRENINGOWA ---
net = MnistLearner(HIDDEN_LAYERS_SIZES, hip_slots=MEMORY_SLOTS)

for epoch in range(EPOCHS):
    epoch_start = time.time()

    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for i in range(0, X_train.shape[0], BATCH_SIZE):
        x_batch = X_shuffled[i:i+BATCH_SIZE]
        y_batch = y_shuffled[i:i+BATCH_SIZE]

        # Wywołujemy metodę WAKE, która sama zdecyduje kiedy spać
        net.train_step_wake(x_batch, y_batch, LEARNING_RATE)

    # Walidacja
    activations = net.forward(X_test)
    out_test = activations[-1]

    preds = np.argmax(out_test, axis=1)
    true_vals = np.argmax(y_test, axis=1)
    acc = np.mean(preds == true_vals) * 100

    print(f"Epoka {epoch+1}/{EPOCHS} | Czas: {time.time()-epoch_start:.2f}s | Dokładność: {acc:.2f}%")

print("-" * 60)
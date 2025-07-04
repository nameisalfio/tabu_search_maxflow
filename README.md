# Maximum Flow Problem - Tabu Search

Implementazione di un algoritmo Tabu Search per il Maximum Flow Problem, sviluppato per il corso di **Heuristics & Metaheuristics**.

## 🎯 Obiettivo

Risolvere il problema del massimo flusso utilizzando una metaeuristica Tabu Search, con l'obiettivo di trovare il flusso massimo che può essere inviato da una sorgente a un pozzo in un grafo diretto con capacità associate agli archi.

## 📁 Struttura Progetto

```
max_flow_tabu/
├── config.yaml
├── data
│   ├── networks
│   │   ├── network_11520.txt
│   │   ├── network_1440.txt
│   │   ├── network_160.txt
│   │   ├── network_23040.txt
│   │   ├── network_2880.txt
│   │   ├── network_4320.txt
│   │   ├── network_500.txt
│   │   ├── network_5760.txt
│   │   ├── network_7200.txt
│   │   └── network_960.txt
│   └── results
├── main.py
├── README.md
├── requirements.txt
├── run_experiments.py
└── src
    ├── __init__.py
    ├── algorithms
    │   ├── __init__.py
    │   └── tabu_search.py
    ├── data
    │   ├── __init__.py
    │   └── network_reader.py
    └── visualization
        ├── __init__.py
        └── plotter.py
```

## 🚀 Quick Start

### Installazione
```bash
# Clona/setup progetto
git clone <repository> && cd max_flow_tabu

# Installa dipendenze
pip install -r requirements.txt

# Aggiungi file network in data/networks/
```

### Esecuzione Singola
```bash
# Esegui su un network specifico
python main.py --network data/networks/network_5760.txt

# Con parametri personalizzati
python main.py --network data/networks/network_5760.txt --config config.yaml
```

### Esperimenti Multipli
```bash
# Multiple run su un network (10 run con semi diversi)
python main.py --network data/networks/network_5760.txt --multiple

# Batch su tutti i network
python run_experiments.py
```

## ⚙️ Configurazione

Il file `config.yaml` permette di personalizzare tutti i parametri:

```yaml
tabu_search:
  max_iterations: 20000      # Numero massimo iterazioni
  tabu_list_size: 10         # Dimensione lista tabu
  aspiration_enabled: true   # Abilita criterio aspirazione
  delta_step: 1             # Passo incremento/decremento

visualization:
  theme: "dark"             # "dark" o "light"
  save_plots: true          # Salva grafici
  show_plots: false         # Mostra grafici
```

## 🧮 Algoritmo Tabu Search

### Pseudocodice

```
TABU_SEARCH_MAX_FLOW(Graph G, source s, sink t)
BEGIN
    // Inizializzazione
    current_solution ← GREEDY_INITIAL_SOLUTION(G, s, t)
    best_solution ← current_solution
    tabu_list ← ∅ (max_size = tabu_list_size)
    
    FOR iteration ← 1 TO max_iterations DO
        // Genera vicinato
        neighbor_moves ← GENERATE_NEIGHBORHOOD(current_solution, G)
        
        // Seleziona migliore mossa ammissibile
        best_move ← null
        best_value ← -∞
        
        FOR each move IN neighbor_moves DO
            IF NOT IS_TABU(move) OR ASPIRATION_CRITERION(move) THEN
                value ← EVALUATE_MOVE(current_solution, move)
                IF value > best_value AND IS_FEASIBLE(move) THEN
                    best_move ← move
                    best_value ← value
                END IF
            END IF
        END FOR
        
        // Applica mossa e aggiorna
        current_solution ← APPLY_MOVE(current_solution, best_move)
        ADD_TO_TABU_LIST(tabu_list, best_move)
        
        IF VALUE(current_solution) > VALUE(best_solution) THEN
            best_solution ← current_solution
        END IF
    END FOR
    
    RETURN best_solution
END
```

### Componenti Principali

1. **Soluzione Iniziale**: Algoritmo greedy basato su Ford-Fulkerson semplificato
2. **Generazione Vicinato**: Mosse di incremento/decremento su archi esistenti
3. **Lista Tabu**: Memoria delle mosse recenti (FIFO con dimensione fissa)
4. **Criterio di Aspirazione**: Permette mosse tabu se migliorano il best-known
5. **Valutazione Fattibilità**: Controllo vincoli di capacità e conservazione flusso

### Operatori di Vicinato

```python
# Mossa di incremento su arco (u,v)
if current_flow[u][v] < capacity[u][v]:
    move = TabuMove((u,v), delta_step, 'increase')

# Mossa di decremento su arco (u,v)  
if current_flow[u][v] > 0:
    move = TabuMove((u,v), delta_step, 'decrease')
```

### Gestione Memoria Tabu

```python
class TabuMove:
    arc: (from_node, to_node)      # Arco modificato
    delta: int                     # Quantità variazione
    move_type: str                 # 'increase' o 'decrease'

# Lista FIFO con dimensione massima
tabu_list = deque(maxlen=tabu_list_size)
```

## 📊 Output e Risultati

### Risultati JSON
```json
{
  "network_file": "network_5760.txt",
  "flow_value": 150,
  "execution_time": 5.23,
  "iterations": 8500,
  "convergence_history": [120, 135, 150, ...],
  "parameters": {...}
}
```

### Grafici Automatici
- **Convergenza**: Evoluzione valore flusso nel tempo
- **Multiple Runs**: Confronto tra diverse esecuzioni  
- **Statistiche**: Confronto tra diversi network
- **Tema Scuro**: Visualizzazioni professionali

### Metriche Calcolate
- **Best**: Miglior valore trovato
- **Mean**: Media su multiple run
- **Std**: Deviazione standard
- **Execution Time**: Tempo medio di esecuzione
- **Iterations**: Numero medio iterazioni

## 🔧 Personalizzazioni

### Modifica Parametri
```python
# In config.yaml
tabu_search:
  max_iterations: 10000    # Per test rapidi
  tabu_list_size: 15       # Più memoria
  delta_step: 2           # Passi maggiori
```

### Cambio Tema Visualizzazioni
```python
# Dark theme (default)
visualization:
  theme: "dark"

# Light theme  
visualization:
  theme: "light"
```

## 📈 Analisi Performance

Il sistema traccia automaticamente:
- Tempo di esecuzione per iterazione
- Storia di convergenza completa
- Statistiche su multiple run
- Confronti tra parametri diversi

## 🧪 Testing

```bash
# Test rapido con parametri ridotti
python main.py --network data/networks/network_5760.txt --config config_quick.yaml

# Debug con output verbose
python main.py --network data/networks/network_5760.txt --multiple
```

## 📚 Riferimenti

- **Problema**: Maximum Flow Problem (Ford-Fulkerson)
- **Metaeuristica**: Tabu Search (Glover, 1986)
- **Applicazioni**: Reti di trasporto, telecomunicazioni, logistica

## ⚠️ Note

- Assicurati che i file network seguano il formato specificato
- Per network grandi, considera di ridurre `max_iterations`
- I grafici sono salvati automaticamente in `data/results/`
- Usa semi fissi per riproducibilità degli esperimenti

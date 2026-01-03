#!/bin/bash

# === Konfiguracja ===
MODEL="qwen2.5-coder:14b"
PROMPT="Napisz kompletną funkcję w Pythonie, która:
1. Wczytuje plik CSV z danymi o sprzedaży,
2. Agreguje dane według miesiąca,
3. Oblicza średnią sprzedaż na produkt,
4. Zapisuje wynik do nowego pliku CSV,
5. Obsługuje błędy wczytywania pliku.
Dołącz komentarze i przykładowe dane wejściowe."
RUNS=3                # Ile razy powtarzamy test
USE_GPU=${1:-1}       # 1 = GPU, 0 = CPU
CPU_THREADS=${2:-0}   # 0 = domyślna liczba wątków CPU

# Plik wyników
RESULT_FILE="ollama_benchmark_results.csv"

# Nagłówek CSV
echo "Run,Time(s),CPU(%),GPU(%),GPU_Mem(MB)" > $RESULT_FILE

for ((i=1;i<=RUNS;i++))
do
    echo "=== Run $i ==="

    # Pomiar startu czasu
    START=$(date +%s)

    # Włączamy GPU/CPU
    if [ "$USE_GPU" -eq 1 ]; then
        OLLAMA_USE_GPU=1
    else
        OLLAMA_USE_GPU=0
    fi

    # Liczba wątków CPU, jeśli ustawiono
    if [ "$CPU_THREADS" -gt 0 ]; then
        export OLLAMA_CPU_THREADS=$CPU_THREADS
    fi

    # Uruchomienie modelu w tle i zapis PID
    ollama run $MODEL "$PROMPT" > /dev/null &
    PID=$!

    # Monitorowanie użycia CPU i GPU
    CPU_USAGE=0
    GPU_USAGE=0
    GPU_MEM=0

    # Pobieramy co 1 sekundę aż zakończy się proces
    while kill -0 $PID 2>/dev/null; do
        # CPU
        CPU_CUR=$(ps -p $PID -o %cpu=)
        CPU_USAGE=$(echo "$CPU_USAGE + $CPU_CUR" | bc)

        # GPU (tylko jeśli jest NVIDIA)
        if command -v nvidia-smi >/dev/null 2>&1; then
            GPU_CUR=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1)
            GPU_MEM_CUR=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
            GPU_USAGE=$(echo "$GPU_USAGE + $GPU_CUR" | bc)
            GPU_MEM=$GPU_MEM_CUR
        fi

        sleep 1
    done

    # Pomiar końca czasu
    END=$(date +%s)
    TIME_ELAPSED=$((END-START))

    # Średnie CPU/GPU
    AVG_CPU=$(echo "$CPU_USAGE / $TIME_ELAPSED" | bc)
    AVG_GPU=$(echo "$GPU_USAGE / $TIME_ELAPSED" | bc)

    # Zapis do CSV
    echo "$i,$TIME_ELAPSED,$AVG_CPU,$AVG_GPU,$GPU_MEM" >> $RESULT_FILE

done

echo "Benchmark zakończony. Wyniki w $RESULT_FILE"

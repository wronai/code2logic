#!/usr/bin/env bash

OUT_DIR="out/code2logic"
mkdir -p "$OUT_DIR"

poetry run code2logic code2logic -f yaml -o "$OUT_DIR/project.c2l.yaml" --compact --with-schema
poetry run code2logic code2logic -f hybrid -o "$OUT_DIR/project.c2l.hybrid.yaml" --with-schema
poetry run code2logic code2logic -f toon -o "$OUT_DIR/project.c2l.toon" --with-schema

echo
echo "=== ROZMIARY WYGENEROWANYCH PLIKÓW (kB) ==="

yaml_bytes=$(stat -c%s "$OUT_DIR/project.c2l.yaml" 2>/dev/null || echo 0)
hybrid_bytes=$(stat -c%s "$OUT_DIR/project.c2l.hybrid.yaml" 2>/dev/null || echo 0)
toon_bytes=$(stat -c%s "$OUT_DIR/project.c2l.toon" 2>/dev/null || echo 0)

yaml_kb=$(awk -v b="$yaml_bytes" 'BEGIN { printf "%.1f", b/1024 }')
hybrid_kb=$(awk -v b="$hybrid_bytes" 'BEGIN { printf "%.1f", b/1024 }')
toon_kb=$(awk -v b="$toon_bytes" 'BEGIN { printf "%.1f", b/1024 }')

echo "YAML:   ${yaml_kb} kB ($OUT_DIR/project.c2l.yaml)"
echo "Hybrid: ${hybrid_kb} kB ($OUT_DIR/project.c2l.hybrid.yaml)"
echo "TOON:   ${toon_kb} kB ($OUT_DIR/project.c2l.toon)"

echo
echo "=== RÓŻNICE ROZMIARÓW (kB) ==="

delta_hybrid_yaml=$(awk -v a="$hybrid_bytes" -v b="$yaml_bytes" 'BEGIN { printf "%.1f", (a-b)/1024 }')
delta_toon_yaml=$(awk -v a="$toon_bytes" -v b="$yaml_bytes" 'BEGIN { printf "%.1f", (a-b)/1024 }')
delta_toon_hybrid=$(awk -v a="$toon_bytes" -v b="$hybrid_bytes" 'BEGIN { printf "%.1f", (a-b)/1024 }')

echo "Hybrid - YAML:  ${delta_hybrid_yaml} kB"
echo "TOON   - YAML:  ${delta_toon_yaml} kB"
echo "TOON   - Hybrid:${delta_toon_hybrid} kB"

poetry run python reproduction_test_code.py
poetry run python multilang_reproduction_tester.py --yaml "$OUT_DIR/project.c2l.yaml" --hybrid "$OUT_DIR/project.c2l.hybrid.yaml" --toon "$OUT_DIR/project.c2l.toon"
poetry run python universal_validator.py --yaml "$OUT_DIR/project.c2l.yaml" --hybrid "$OUT_DIR/project.c2l.hybrid.yaml" --toon "$OUT_DIR/project.c2l.toon" --language python --verbose


poetry run code2logic code2logic -f compact --function-logic -o "$OUT_DIR/project.func.logicml"
poetry run code2logic code2logic -f compact --function-logic -o "$OUT_DIR/project.func.json"
poetry run code2logic code2logic -f toon --function-logic -o "$OUT_DIR/project.func.yaml"
poetry run code2logic code2logic -f json --function-logic -o "$OUT_DIR/project.func.toon"
poetry run code2logic code2logic -f compact -q --function-logic "$OUT_DIR/project.func.logicml"

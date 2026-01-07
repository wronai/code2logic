# examples/code2logic

Ten przykład pokazuje jak uruchomić **code2logic** na małym projekcie Python i wygenerować plik `.c2l.yaml` (oraz opcjonalnie inne formaty).

## Co jest w środku

- `sample_project/` – mini-projekt Python do analizy
- `out/` – katalog na wyniki
- `Dockerfile` + `docker-compose.yml` – uruchamianie w kontenerze (z tego katalogu)

## Uruchomienie lokalnie (z tego katalogu)

W repozytorium nie musisz nic instalować globalnie – wystarczy dodać root repo do `PYTHONPATH`.

```bash
PYTHONPATH=../.. python -m code2logic sample_project \
  -f yaml \
  --no-install \
  -o out/project.c2l.yaml
```

Dodatkowe warianty:

```bash
# Flat JSON (przydatne do porównań)
PYTHONPATH=../.. python -m code2logic sample_project \
  -f json --flat \
  --no-install \
  -o out/project.c2l.json

# CSV (często najlepsze dla LLM)
PYTHONPATH=../.. python -m code2logic sample_project \
  -f csv \
  --no-install \
  -o out/project.c2l.csv
```

Jeśli chcesz, możesz usunąć `--no-install` – wtedy `code2logic` spróbuje sam doinstalować brakujące opcjonalne zależności.

## Uruchomienie w Dockerze (z tego katalogu)

```bash
docker compose run --rm code2logic
```

Wynik pojawi się w:

- `out/project.c2l.yaml`

## Następne kroki

- Wygenerowany plik możesz wykorzystać w:
  - `../logic2test` (generowanie testów)
  - `../logic2code` (generowanie kodu)

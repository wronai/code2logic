# examples/logic2code

Ten przykład pokazuje jak uruchomić **logic2code** na pliku wyjściowym z `code2logic` (YAML/Hybrid/TOON) i wygenerować szkielety kodu źródłowego.

## Co jest w środku

- `input/sample_project.c2l.yaml` – mały plik logiczny (format zgodny z parserem `logic2code`)
- `out/` – katalog na wygenerowany kod
- `Dockerfile` + `docker-compose.yml` – uruchamianie w kontenerze (z tego katalogu)

## Uruchomienie lokalnie (z tego katalogu)

```bash
PYTHONPATH=../.. python -m logic2code input/sample_project.c2l.yaml \
  -o out/generated_code
```

Warianty:

```bash
# Tylko stuby (zamiast NotImplementedError)
PYTHONPATH=../.. python -m logic2code input/sample_project.c2l.yaml \
  -o out/generated_code \
  --stubs-only

# Podgląd bez generowania
PYTHONPATH=../.. python -m logic2code input/sample_project.c2l.yaml --summary
```

## Uruchomienie w Dockerze (z tego katalogu)

```bash
docker compose run --rm logic2code
```

Wyniki pojawią się w:

- `out/generated_code/`

## Workflow: code2logic -> logic2code

Jeśli chcesz użyć realnego projektu zamiast sample-a, możesz najpierw wygenerować plik `.c2l.yaml` w `../code2logic`, a potem wskazać go tutaj jako `input`.

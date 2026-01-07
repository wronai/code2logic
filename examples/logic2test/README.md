# examples/logic2test

Ten przykład pokazuje jak uruchomić **logic2test** na pliku wyjściowym z `code2logic` (YAML/Hybrid/TOON) i wygenerować szkielety testów.

## Co jest w środku

- `input/sample_project.c2l.yaml` – mały plik logiczny (format zgodny z parserem `logic2test`)
- `out/` – katalog na wygenerowane testy
- `Dockerfile` + `docker-compose.yml` – uruchamianie w kontenerze (z tego katalogu)

## Uruchomienie lokalnie (z tego katalogu)

```bash
PYTHONPATH=../.. python -m logic2test input/sample_project.c2l.yaml \
  -o out/tests \
  --type all
```

Podgląd bez generowania:

```bash
PYTHONPATH=../.. python -m logic2test input/sample_project.c2l.yaml --summary
```

## Uruchomienie w Dockerze (z tego katalogu)

```bash
docker compose run --rm logic2test
```

Wyniki pojawią się w:

- `out/tests/`

## Workflow: code2logic -> logic2test

Jeśli chcesz użyć realnego projektu zamiast sample-a, możesz najpierw wygenerować plik `.c2l.yaml` w `../code2logic`, a potem wskazać go tutaj jako `input`.

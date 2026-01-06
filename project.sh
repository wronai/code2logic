poetry run code2logic code2logic -f yaml -o project.c2l.yaml --compact --with-schema
poetry run code2logic code2logic -f yaml -o project.c2l.hybrid.yaml --hybrid --with-schema
poetry run code2logic code2logic -f toon -o project.c2l.toon --with-schema

poetry run python reproduction_test_code.py
poetry run python multilang_reproduction_tester.py --yaml project.c2l.yaml --hybrid project.c2l.hybrid.yaml --toon project.c2l.toon
poetry run python universal_validator.py --yaml project.c2l.yaml --hybrid project.c2l.hybrid.yaml --toon project.c2l.toon --language python --verbose
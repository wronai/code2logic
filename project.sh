#!/usr/bin/env bash

poetry run code2logic code2logic -f yaml -o project.c2l.yaml --compact --with-schema
poetry run code2logic code2logic -f yaml -o project.c2l.hybrid.yaml --hybrid --with-schema
poetry run code2logic code2logic -f toon -o project.c2l.toon --with-schema

poetry run python reproduction_test_code.py
poetry run python multilang_reproduction_tester.py --yaml project.c2l.yaml --hybrid project.c2l.hybrid.yaml --toon project.c2l.toon
poetry run python universal_validator.py --yaml project.c2l.yaml --hybrid project.c2l.hybrid.yaml --toon project.c2l.toon --language python --verbose


poetry run code2logic code2logic -f compact --function-logic -o project.func.logicml
poetry run code2logic code2logic -f compact --function-logic -o project.func.json
poetry run code2logic code2logic -f toon --function-logic -o project.func.yaml
poetry run code2logic code2logic -f json --function-logic -o project.func.toon
poetry run code2logic code2logic -f compact -q --function-logic project.func.logicml

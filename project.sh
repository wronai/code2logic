code2logic code2logic -f yaml -o project.c2l.yaml --compact --with-schema
code2logic code2logic -f yaml -o project.c2l.hybrid.yaml --hybrid --with-schema
code2logic code2logic -f toon -o project.c2l.toon --compact --with-schema
python reproduction_test_code.py
python multilang_reproduction_tester.py
python universal_validator.py
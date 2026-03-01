#!/usr/bin/env bash
pip install code2logic --upgrade
#code2logic ./ -f toon --compact --no-repeat-module --function-logic --with-schema --name project -o ./
code2logic ./ -f toon --compact --name project -o ./

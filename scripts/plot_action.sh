#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
exec python3 -m neural_vector_field.action_graph "$@"

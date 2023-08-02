#!/bin/bash

python plot/plot_fig12.py results  --subset n1-sharegpt --duration 600
python plot/plot_fig12.py results  --subset n1-alpaca --duration 600
python plot/plot_fig14.py results --subset parallel --duration 600
python plot/plot_fig14.py results --subset beam --duration 600
python plot/plot_fig16.py prefix_exp
python plot/plot_fig17.py results

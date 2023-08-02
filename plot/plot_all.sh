#!/bin/bash

python plot/plot_fig2.py
python plot/plot_fig12.py results  --subset n1-sharegpt --duration 600
python plot/plot_fig12.py results  --subset n1-alpaca --duration 600
python plot/plot_fig13.py
python plot/plot_fig14.py results --subset parallel --duration 600
python plot/plot_fig14.py results --subset beam --duration 600
python plot/plot_fig15.py
python plot/plot_fig16.py prefix_exp
python plot/plot_fig17.py results
python plot/plot_fig18_b.py
python plot/plot_fig19_a.py
python plot/plot_fig19_b.py

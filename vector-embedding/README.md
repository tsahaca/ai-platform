Created generation_params_demo.py. It uses GPT-2 (small, no auth required) to run 9 scenarios side-by-side:

Scenario	What it shows
Greedy	Deterministic baseline
Temperature 0.3 / 1.0 / 1.5	Effect of sharpening vs. flattening the distribution
top-k=10 / top-k=50	Restricting candidate tokens at each step
top-p=0.7 / top-p=0.95	Nucleus (probability-mass) filtering
Combined t=0.8, k=40, p=0.9	Typical production setting
All scenarios use the same seed so the differences are purely from the parameters. Run it with:
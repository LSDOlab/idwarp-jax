# Half-cube stretch

From the repository root:

```bash
python -m pip install -e .
python simple_example/cube_stretch.py
```

Change `DTYPE` and `DEVICE` near the top of the script to choose
`"f32"`/`"f64"` and `"cpu"`/`"gpu"`. The script prints a
VJP/finite-difference check and writes `simple_example/cube_stretch.png`.

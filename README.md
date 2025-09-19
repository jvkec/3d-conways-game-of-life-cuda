# 3D Conway's Game of Life (CUDA + OpenGL)

A CUDA implementation of 3D Conway-like cellular automata with a lightweight OpenGL viewer. 
Use the batch runner on a GPU machine to generate states, then visualize them on macOS.

## Screenshots

![Screenshot of the 3D Conway's Game of Life simulation](assets/images/Screenshot%202025-09-18%20at%2010.57.56%E2%80%AFPM.png)
![Screenshot of the 3D Conway's Game of Life simulation](assets/images/Screenshot%202025-09-18%20at%2010.58.06%E2%80%AFPM.png)
![Screenshot of the 3D Conway's Game of Life simulation](assets/images/Screenshot%202025-09-18%20at%2010.58.22%E2%80%AFPM.png)
![Screenshot of the 3D Conway's Game of Life simulation](assets/images/Screenshot%202025-09-18%20at%2010.58.31%E2%80%AFPM.png)
![Screenshot of the 3D Conway's Game of Life simulation](assets/images/Screenshot%202025-09-18%20at%2010.58.41%E2%80%AFPM.png)

## What's here

- CUDA kernels and a simple driver to run batches and save states.
- A standalone 3D viewer that loads saved runs and plays them back.
- Binary state format with a small header and RLE-compressed boolean grid.

## Build

On macOS with Homebrew:

- Dependencies: `glfw`, `glew`. Install via Homebrew if needed:
  - `brew install glfw glew`

Then build from the repo root:

```
make viewer_3d       # OpenGL playback app
make batch_runner    # CUDA batch runner (build on a CUDA-capable machine)
```

## Running the viewer

The viewer expects a directory of `.bin` state files. By default it looks under `runs/`:

```
./viewer_3d                 # loads runs/massive_growth or runs/states if present
./viewer_3d runs/states     # or point it at any directory of .bin files
```

### Controls

- Space: play/pause
- Left/Right: previous/next frame
- Mouse drag: orbit
- Scroll: zoom
- R: reset camera

## Generating runs (batch mode)

Run batches on a CUDA machine:

```
make batch_runner \
  --grid 96x96x96 \
  --generations 300 \
  --save-every 10 \
  --output runs/states \
  --rules B_MIN,B_MAX,S_MIN,S_MAX \
  --density 0.25
```

**Notes:**

- Default output directory is `runs/states`.
- `--rules` maps to the 3D Moore neighborhood (26 neighbors). Examples that grow interesting structures:
  - `6,6,5,7`  (B6 / S5–7)
  - `5,7,6,6`  (B5–7 / S6)
  - The shipped defaults are stricter (14–19) and can look dense with high initial density.

## File format

- Each `.bin` is: header + RLE data.
  - Header includes: magic ("CGOL"), version, width/height/depth, generation, timestamp, rules, data_size, checksum.
  - Data is run-length encoded pairs: `(run:uint8, value:uint8)` repeating until `data_size` bytes.
- The viewer also supports a legacy format of raw `(x,y,z)` triplets (int32), optionally prefixed with a count.

## Tips for clearer visuals

- Early frames with large populations can look like a solid block. If that happens:
  - Lower alpha in `Renderer3D::render()` (e.g., 0.05–0.10).
  - Shrink cube size in `Renderer3D::createCubeGeometry()` (e.g., ±0.25 instead of ±0.35).
  - Try rules/density that don't immediately fill the volume.

## Repo layout

- `src/cuda/` — kernels and batch utilities
- `include/cuda/` — CUDA headers and types
- `src/opengl/` — viewer code (camera, renderer, GL setup)
- `include/opengl/` — viewer headers
- `runs/` — where new run data is written by default (you can change this)

## Troubleshooting

- If the viewer opens to a uniform green block, you probably loaded a dense frame. Use the tips above, or step frames with arrow keys.
- If the viewer loads but shows just a few cubes, your files are likely in the wrong format; ensure you're using the current `.bin` format produced by `batch_runner`.
- On macOS the viewer uses OpenGL 4.1 core (Metal-backed). The shaders are embedded and target GLSL 410.

## License

MIT

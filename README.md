# 3×3 cube with a neural network solver

A 3D cube you can scramble, turn and solve in the browser. The Solve button asks a neural network, trained on random scrambles, for a solution and plays it back. The cube's state is kept in a module separate from the drawing code, so the page, the solver and the tests all share one cube model.

Try it at [joshcc27.github.io/rubiks-cube-solver](https://joshcc27.github.io/rubiks-cube-solver/).

![A scrambled cube with the network's 27-move solution paused after move 9](docs/screenshot.png)

## What works now

- All 18 quarter turns in standard notation: U, D, R, L, F, B, the slice turns M, E, S, and their primes. Directions follow the usual convention (U, R, F clockwise seen from that face; D, L, B clockwise seen from their own side; M follows L, E follows D, S follows F).
- Turning a layer by dragging a sticker. The turn is picked from the drag direction relative to the current camera view, so it works after orbiting to any angle.
- Keyboard turns.
- A 20-move scramble using face turns only, with no two consecutive moves on the same axis. Scramble resets the cube first, so the scramble string plus the move list (and, while a solution is showing, the part of it already played) always describe the state from solved.
- Solve, which runs the trained network in the browser and plays its solution. It works from any state the page can reach, including after slice turns. While it searches, the button reads Stop and the status shows how deep the search is and for how long it has run. The first Solve shows the network's download progress.
- A solution player: the network's moves listed on their own, with the current one highlighted, play and pause, step back and forward, jump to the start or end or to any move by clicking it, and a playback speed from 0.5× to 3×.
- Typing or pasting moves, such as `R U R' U'` or `RUR'U'`, to apply them to the cube or to use them as the scramble.
- Face labels on the centre stickers, a Reset view button, and an About dialog that explains the solver.
- Undo, a move counter and the list of moves since the scramble. While a solution is showing, its moves are listed in the player instead, and Undo steps back through the part already played. If the cube is then changed some other way, the played moves join the move list, so Undo can still take them back.
- Solved detection, which checks that every face shows one colour.
- A layout for phones: the cube fits the screen, help opens from a ? button and explains the touch controls, and the turn buttons fold away.

## Controls

| Input | Action |
| --- | --- |
| Drag a sticker | Turn that sticker's layer in the drag direction |
| Drag the background | Orbit the camera |
| Mouse wheel | Zoom |
| `U` `D` `R` `L` `F` `B` `M` `E` `S` | Clockwise turn |
| `Shift` + letter | Counter-clockwise (prime) turn |
| `Ctrl` + `Z` | Undo the last turn, or step back through a solution |
| `←` `→` | Step back or forward through a solution |
| `Space` | Play or pause a solution |
| `Esc` | Close the help on a phone-sized screen |
| Buttons | Scramble, Solve (Stop while searching), Undo, Reset, the solution player, and single turns under Turn buttons |

Letter keys are read without regard to case, so Caps Lock does not change the direction. Turns made while another turn is animating are queued. Scramble, Solve and Undo are disabled until the queue is empty and while a solution plays. Reset works at any time and clears the queue. Turning the cube, Undo, Scramble or Reset while the solver is searching cancels the search, and any change to the cube other than through the player puts the solution away.

The page remembers a few per-browser settings: whether the help, labels and turn buttons are shown, and the playback speed.

## Run locally

The page uses ES modules, so opening `index.html` directly won't work. Serve the folder with any static server instead. From the repository root:

```sh
python -m http.server 8000   # then open http://localhost:8000/
npx serve .                  # needs Node; prints the URL to open
```

There is no build step and nothing to install. Three.js and ONNX Runtime Web are loaded from jsDelivr, so the page needs network access the first time it loads. The first Solve downloads about 12 MB: the network (`models/policy.onnx`, 9 MB) and the ONNX Runtime WebAssembly engine.

## Tests

The cube model and the solver's search have tests that use Node's built-in test runner (Node 18 or newer, no dependencies):

```sh
node --test      # or: npm test
```

They check that:

- any move done four times returns the cube to where it started
- any move followed by its inverse does the same
- `(R U R' U')` done six times is the identity
- a double turn equals two quarter turns
- each turn sends stickers where standard notation says it should (after U, the front top row shows the right face's colour)
- scrambles never turn the same axis twice in a row
- relabelling by centres undoes what slice moves do to the centres, and face turns commute with it
- typed moves are read with or without spaces, and anything that isn't a move gives a readable error
- `training/moves.json` is up to date with `cube-model.js`
- the search, run with a stand-in for the network, solves every two-move state and cubes turned with slice moves, and stops when cancelled
- the page's record of moves always explains the cube on screen: through user moves and Undo, stepping and jumping through a solution, and putting a part-played solution away, even with a step still animating

## The neural network solver

**How it learns.** Training scrambles solved cubes with random face turns. Each state along the way is labelled with the move that undoes the turn just made. The network sees the state and learns to predict that move, so its output is a probability for each of the 18 face turns being a step back towards solved. Scrambles are generated on the fly, so there is no dataset. The method follows Takano, *Self-Supervision is All You Need for Solving Rubik's Cube* (2023).

**How it solves.** Always taking the network's top move solves short scrambles but rarely a random cube. Beam search does better: it keeps the 512 move sequences with the highest total probability and extends them one move at a time until one reaches solved. If none does within 60 moves, the page tries again with 2048. In two test runs of 50 random cubes each, a width of 512 solved all 100, averaging 31 and 33 moves. A random cube can usually be solved in about 18, so the network's solutions are valid but long.

**In the page.** The network runs with ONNX Runtime Web on the CPU through WebAssembly, in a Web Worker so the page keeps animating while it searches (`solver-worker.js`). A search takes about 2 to 3 seconds on a laptop. The search itself is in `solver.js` and matches the Python version in `training/src/search.py`.

**Shared moves.** The Python code doesn't define its own moves. `scripts/export-moves.mjs` writes the 18 face-turn permutations from `cube-model.js` to `training/moves.json`, along with example scrambles that the Python tests replay. A Node test fails if the file is out of date, so rerun `npm run export-moves` after changing the model. Slice moves move the centres, so the page passes its state through `relabelByCentres` first. After that, face turns alone can solve it.

### Training

The training code is in `training/`. It uses Python 3.12, PyTorch and [uv](https://docs.astral.sh/uv/). From `training/`:

```sh
uv sync
uv run python -m pytest                       # tests, including a short training run and the ONNX export
uv run python -m src.train --minutes 60       # writes checkpoints/policy.pt
uv run python -m src.evaluate checkpoints/policy.pt --depths 5 10 15 20 100 --widths 1 64 512
uv run python -m src.solve checkpoints/policy.pt "R U R' U' F2 D L"
uv run python -m src.export_onnx checkpoints/policy.pt ../models/policy.onnx   # update the page's network
```

`src.solve` reads scrambles the way the page does, slice turns and moves without spaces included, or the page's 54-letter facelet string, and rejects anything it can't read with a clear message. `src.train --help` lists the settings: network size, batch size, time or step budget, device and CPU threads. Training picks a CUDA or Intel GPU when PyTorch can see one, and otherwise runs on the CPU. On a CPU with performance and efficiency cores, setting `--threads` to the number of performance cores can be noticeably faster.

The network in `models/policy.onnx` is a fully connected network with four residual blocks (2.3M parameters), trained for one hour on a laptop CPU (38.7M examples). Its ONNX file takes the one-hot colours of the 48 non-centre stickers and returns log-probabilities for the 18 face turns.

## How it is built

- `cube-model.js` holds the logical cube: a 54-entry array of face letters in the order U, R, F, D, L, B, nine stickers per face. For a solved cube `state.join('')` gives the usual facelet string `UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB`. Each move is a sticker permutation computed once at load time by rotating sticker positions and normals. The module has no dependency on Three.js, so the page, the tests and the solver all use the same code.
- `main.js` draws the cube with Three.js 0.165, loaded through an import map in `index.html`, and handles input. The 27 small cubes never really move: a turn animates one layer, then snaps it back and recolours the stickers from the model. They all share one geometry and seven materials. Solve sends the state to the worker. The solution player feeds the solution into the same turn queue one move at a time, so pausing takes effect after the current move, and jumping recomputes the state from the model without animating. The face labels are children of the centre cubies, so they stay on their faces through slice turns.
- `session.js` holds the cube on screen and the record of how it got there: the scramble, the moves since, and the network's solution while it is showing, with what Undo, stepping and jumping do to them. It has no rendering or DOM, so its rules are tested in Node.
- `solver.js` is the beam search, and `solver-worker.js` runs it with the network in a Web Worker.
- `index.html` is the page layout and styles. Wide screens get the cube beside a side panel; narrow ones stack them.
- `training/` holds the Python code that trains, evaluates and exports the network.

## Roadmap

- Train for longer, ideally on a GPU, to shorten the solutions.
- Enter a real cube's colours and solve it, with checks that the colours describe a cube that can exist.

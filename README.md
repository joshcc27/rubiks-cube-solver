# 3×3 cube (solver in progress)

**Status:** a 3D cube you can scramble and turn in the browser. It can't solve itself yet; a solver is planned.

I built this as the front end for a cube solver. The cube, the controls and the move logic are done, and the logical model is set up so a solver can read the cube's state and play its solution back.

**Try it:** [joshcc27.github.io/rubiks-cube-solver](https://joshcc27.github.io/rubiks-cube-solver/)

![The cube after a scramble and three turns](docs/screenshot.png)

## What works now

- All 18 quarter turns in standard notation: U, D, R, L, F, B, the slice turns M, E, S, and their primes. Directions follow the usual convention (U, R, F clockwise seen from that face; D, L, B clockwise seen from their own side; M follows L, E follows D, S follows F).
- Turning a layer by dragging a sticker. The turn is picked from the drag direction relative to the current camera view, so it works after orbiting to any angle.
- Keyboard turns.
- A 20-move scramble using face turns only, with no two consecutive moves on the same axis. Scramble resets the cube first, so the scramble string plus the move list always describe the state from solved.
- Undo, a move counter and the list of moves made since the scramble.
- Solved detection, which checks that every face shows one colour.

## Controls

| Input | Action |
| --- | --- |
| Drag a sticker | Turn that sticker's layer in the drag direction |
| Drag the background | Orbit the camera |
| Mouse wheel | Zoom |
| `U` `D` `R` `L` `F` `B` `M` `E` `S` | Clockwise turn |
| `Shift` + letter | Counter-clockwise (prime) turn |
| `Ctrl` + `Z` | Undo the last turn |
| Buttons | Single turns, Scramble, Undo, Reset |

Letter keys are read without regard to case, so Caps Lock does not change the direction. Turns made while another turn is animating are queued. Scramble and Undo are disabled until the queue is empty. Reset works at any time and clears the queue.

## Run locally

The page uses ES modules, so opening `index.html` directly won't work. Serve the folder with any static server instead. From the repository root:

```sh
python -m http.server 8000   # then open http://localhost:8000/
npx serve .                  # needs Node; prints the URL to open
```

There is no build step and nothing to install. Three.js is loaded from jsDelivr, so the page needs network access the first time it loads.

## Tests

The cube model has tests that use Node's built-in test runner (Node 18 or newer, no dependencies):

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

## How it is built

- `cube-model.js` holds the logical cube: a 54-entry array of face letters in the order U, R, F, D, L, B, nine stickers per face. For a solved cube `state.join('')` gives the usual facelet string `UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB`. Each move is a sticker permutation computed once at load time by rotating sticker positions and normals. The module has no dependency on Three.js, so the page, the tests and a future solver all use the same code.
- `main.js` draws the cube with Three.js 0.165, loaded through an import map in `index.html`, and handles input. The 27 small cubes never really move: a turn animates one layer, then snaps it back and recolours the stickers from the model. They all share one geometry and seven materials.
- `index.html` is the page layout and styles.

## Roadmap

- Add a solver that reads the facelet array from `cube-model.js` and plays the solution back as an animated move sequence.
- Show the solution moves and step through them.

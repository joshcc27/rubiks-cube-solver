import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FACE_MOVES, solvedState, applyMoves, isSolved, relabelByCentres } from '../cube-model.js';
import { beamSearch, encode, INPUT_SIZE, SearchAborted } from '../solver.js';

const solved = solvedState();

// Stand-in for the network: every move equally likely.
const uniform = (_encoded, n) => new Float32Array(n * 18).fill(-Math.log(18));

test('encoding is one colour per non-centre sticker', () => {
  const states = Uint8Array.from(solved.join('') + applyMoves(solved, "R U").join(''), c => 'URFDLB'.indexOf(c));
  const x = encode(states, 2);
  assert.equal(x.length, 2 * INPUT_SIZE);
  for (let s = 0; s < 2; s++) {
    let total = 0;
    for (let i = 0; i < INPUT_SIZE; i++) total += x[s * INPUT_SIZE + i];
    assert.equal(total, 48);
  }
  // Sticker 0 (U1) shows U, colour 0, so the first slot is set.
  assert.equal(x[0], 1);
});

test('a solved cube needs no moves', async () => {
  assert.deepEqual(await beamSearch(solved, uniform, { width: 4 }), []);
});

test('a beam wide enough to be exhaustive solves every two-move state', async () => {
  for (const a of FACE_MOVES) {
    for (const b of FACE_MOVES) {
      if (a[0] === b[0]) continue;
      const state = applyMoves(solved, [a, b]);
      const path = await beamSearch(state, uniform, { width: 18, maxDepth: 2 });
      assert.ok(path && path.length === 2, `${a} ${b}`);
      assert.ok(isSolved(applyMoves(state, path)), `${a} ${b} -> ${path.join(' ')}`);
    }
  }
});

test('a cube turned with slice moves is solved after relabelling', async () => {
  // M E' is four face turns away once relabelled. There are at most 3645
  // three-move sequences, so this width checks every four-move one.
  const state = applyMoves(solved, "M E'");
  const path = await beamSearch(relabelByCentres(state), uniform, { width: 4000, maxDepth: 4 });
  assert.ok(path && path.length === 4);
  assert.ok(isSolved(applyMoves(state, path)));
});

test('the search rejects a state whose centres are not home', async () => {
  await assert.rejects(beamSearch(applyMoves(solved, 'M'), uniform), /centres must be home/);
});

test('an aborted search throws SearchAborted', async () => {
  const signal = { aborted: true };
  await assert.rejects(beamSearch(applyMoves(solved, "R U F"), uniform, { signal }), SearchAborted);
});

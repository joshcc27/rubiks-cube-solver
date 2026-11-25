// Run with: node --test   (Node 18 or newer, no install step)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  BASE_MOVES, solvedState, applyMove, applyMoves, invertMove, isSolved, generateScramble,
} from '../cube-model.js';

const ALL_MOVES = Object.keys(BASE_MOVES).flatMap(b => [b, `${b}'`, `${b}2`]);
const solved = solvedState();

// A scrambled starting point, so the identity checks are not only run on a
// solved cube (where some wrong permutations could still look fine).
const start = applyMoves(solved, "R U2 F' L D B2 M E' S");

const row = (state, face, r) => {
  const base = 'URFDLB'.indexOf(face) * 9 + r * 3;
  return state.slice(base, base + 3).join('');
};
const col = (state, face, c) => {
  const base = 'URFDLB'.indexOf(face) * 9 + c;
  return [state[base], state[base + 3], state[base + 6]].join('');
};

test('solved state is solved and has the standard facelet string', () => {
  assert.equal(solved.join(''), 'U'.repeat(9) + 'R'.repeat(9) + 'F'.repeat(9) + 'D'.repeat(9) + 'L'.repeat(9) + 'B'.repeat(9));
  assert.ok(isSolved(solved));
  assert.ok(!isSolved(start));
});

test('every move applied four times is the identity', () => {
  for (const mv of ALL_MOVES) {
    assert.deepEqual(applyMoves(start, [mv, mv, mv, mv]), start, mv);
  }
});

test('every move followed by its inverse is the identity', () => {
  for (const mv of ALL_MOVES) {
    assert.deepEqual(applyMove(applyMove(start, mv), invertMove(mv)), start, mv);
  }
});

test('X2 equals X X', () => {
  for (const b of Object.keys(BASE_MOVES)) {
    assert.deepEqual(applyMove(start, `${b}2`), applyMoves(start, [b, b]), b);
  }
});

test("(R U R' U') six times is the identity", () => {
  const seq = "R U R' U' ".repeat(6);
  assert.deepEqual(applyMoves(start, seq), start);
  assert.notDeepEqual(applyMoves(start, "R U R' U' ".repeat(3)), start);
});

// Sticker checks against standard notation. "F gets R" means the front face
// row now shows the colour that was on the right face.
test('U: front top row takes the right face colours, and moves on to the left face', () => {
  const s = applyMove(solved, 'U');
  assert.equal(row(s, 'F', 0), 'RRR');
  assert.equal(row(s, 'L', 0), 'FFF');
  assert.equal(row(s, 'B', 0), 'LLL');
  assert.equal(row(s, 'R', 0), 'BBB');
});

test('D: front bottom row takes the left face colours', () => {
  const s = applyMove(solved, 'D');
  assert.equal(row(s, 'F', 2), 'LLL');
  assert.equal(row(s, 'R', 2), 'FFF');
});

test('E follows D', () => {
  const s = applyMove(solved, 'E');
  assert.equal(row(s, 'F', 1), 'LLL');
  assert.equal(row(s, 'R', 1), 'FFF');
});

test('R: up face right column takes the front colours', () => {
  const s = applyMove(solved, 'R');
  assert.equal(col(s, 'U', 2), 'FFF');
  assert.equal(col(s, 'B', 0), 'UUU');
  assert.equal(col(s, 'F', 2), 'DDD');
});

test('L and M: front left/middle column takes the up colours', () => {
  assert.equal(col(applyMove(solved, 'L'), 'F', 0), 'UUU');
  assert.equal(col(applyMove(solved, 'M'), 'F', 1), 'UUU');
});

test('F: right face left column takes the up colours', () => {
  const s = applyMove(solved, 'F');
  assert.equal(col(s, 'R', 0), 'UUU');
  assert.equal(row(s, 'U', 2), 'LLL');
});

test('S follows F', () => {
  const s = applyMove(solved, 'S');
  assert.equal(col(s, 'R', 1), 'UUU');
  assert.equal(row(s, 'U', 1), 'LLL');
});

test('B: left face left column takes the up colours', () => {
  const s = applyMove(solved, 'B');
  assert.equal(col(s, 'L', 0), 'UUU');
  assert.equal(row(s, 'U', 0), 'RRR');
});

test('scrambles use face turns only and never repeat an axis back to back', () => {
  const axis = { U: 'y', D: 'y', R: 'x', L: 'x', F: 'z', B: 'z' };
  for (let n = 0; n < 200; n++) {
    const scr = generateScramble(20);
    assert.equal(scr.length, 20);
    for (let i = 0; i < scr.length; i++) {
      assert.match(scr[i], /^[UDRLFB]['2]?$/);
      if (i > 0) assert.notEqual(axis[scr[i][0]], axis[scr[i - 1][0]], scr.join(' '));
    }
  }
});

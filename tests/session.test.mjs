import { test } from 'node:test';
import assert from 'node:assert/strict';
import { solvedState, applyMoves, invertMove, isSolved } from '../cube-model.js';
import { CubeSession } from '../session.js';

// Play turns the way main.js does: each one is recorded once it has "finished animating".
const play = (session, turns) => turns.forEach(({ move, kind }) => session.applyTurn(move, kind));
const user = moves => moves.split(' ').map(move => ({ move, kind: 'user' }));

// The record must always explain the cube once no turn is animating.
function assertRecordMatches(session, message) {
  assert.deepEqual(applyMoves(solvedState(), session.movesFromSolved()), session.state, message);
}

function scrambled(moves = "R U F' L2 D B'") {
  const session = new CubeSession();
  session.startScramble(moves.split(' '));
  play(session, session.scramble.map(move => ({ move, kind: 'scramble' })));
  return session;
}

// A solution that really solves the current state: every move so far, undone.
const solutionFor = session => session.movesFromSolved().slice().reverse().map(invertMove);

test('user moves and undo keep the record', () => {
  const s = scrambled();
  play(s, user("F R2"));
  assert.deepEqual(s.history, ['F', 'R2']);
  play(s, [s.undoTurn()]);
  assert.deepEqual(s.history, ['F']);
  assertRecordMatches(s);
});

test('stepping through a solution moves pos and Undo steps back through it', () => {
  const s = scrambled();
  s.showSolution(solutionFor(s), 1.2);
  for (let i = 0; i < 3; i++) play(s, [s.nextSolutionTurn()]);
  assert.equal(s.solution.pos, 3);
  assert.equal(s.undoTurn().kind, 'solve-back');
  play(s, [s.undoTurn()]);
  assert.equal(s.solution.pos, 2);
  assert.deepEqual(s.history, []);
  assertRecordMatches(s);
});

test('playing a whole solution solves the cube, and there is no step past either end', () => {
  const s = scrambled();
  s.showSolution(solutionFor(s), 1);
  assert.equal(s.prevSolutionTurn(), null);
  let turn;
  while ((turn = s.nextSolutionTurn())) play(s, [turn]);
  assert.ok(isSolved(s.state));
  assertRecordMatches(s);
});

test('putting a part-played solution away keeps its moves, so Undo never invents a state', () => {
  // The review's case: scramble, F, solve, pause after 5 moves, then R.
  const s = scrambled();
  play(s, user('F'));
  s.showSolution(solutionFor(s), 1);
  for (let i = 0; i < 5; i++) play(s, [s.nextSolutionTurn()]);
  const afterFive = s.state.slice();
  s.dropSolution(); // what main.js does when the person turns the cube
  play(s, user('R'));
  assertRecordMatches(s);

  play(s, [s.undoTurn()]); // takes back R
  assert.deepEqual(s.state, afterFive);
  play(s, [s.undoTurn()]); // takes back the network's fifth move, not F
  assertRecordMatches(s);
  // Undo all the way back reaches the scrambled cube, through states that were on screen.
  let turn;
  while ((turn = s.undoTurn())) play(s, [turn]);
  assert.deepEqual(s.state, applyMoves(solvedState(), s.scramble));
});

test('a solution step still animating when the solution is put away still counts', () => {
  const s = scrambled();
  s.showSolution(solutionFor(s), 1);
  play(s, [s.nextSolutionTurn()]);
  play(s, [s.nextSolutionTurn()]);
  const inFlight = s.nextSolutionTurn();
  s.dropSolution();
  play(s, [inFlight, ...user('U')]);
  assert.equal(s.history.length, 4);
  assertRecordMatches(s);

  const back = scrambled();
  back.showSolution(solutionFor(back), 1);
  play(back, [back.nextSolutionTurn()]);
  play(back, [back.nextSolutionTurn()]);
  const stepBack = back.prevSolutionTurn();
  back.dropSolution();
  play(back, [stepBack]);
  assert.equal(back.history.length, 1);
  assertRecordMatches(back);
});

test('jumping sets the state from the solution without animating', () => {
  const s = scrambled();
  s.showSolution(solutionFor(s), 1);
  s.jumpTo(4);
  assert.equal(s.solution.pos, 4);
  assert.equal(s.solution.playing, false);
  assertRecordMatches(s);
  s.jumpTo(99);
  assert.ok(isSolved(s.state));
  s.jumpTo(-1);
  assert.equal(s.solution.pos, 0);
  assertRecordMatches(s);
});

test('a new scramble or a reset clears the record', () => {
  const s = scrambled();
  s.showSolution(solutionFor(s), 1);
  s.reset();
  assert.deepEqual([s.scramble, s.history, s.solution], [[], [], null]);
  assert.ok(isSolved(s.state));
  assert.equal(s.undoTurn(), null);
});

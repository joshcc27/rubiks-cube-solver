// The cube's state and the record of how it got there: the scramble, the
// moves made since, and the network's solution while one is showing. There
// is no rendering or DOM here, so these rules are tested in Node
// (tests/session.test.mjs). main.js animates each turn, then calls applyTurn,
// so a turn that is still animating is not yet part of the record.
//
// Once no turn is animating, the record always explains the cube on screen:
//   state = solved + scramble + history + solution.moves[0 .. solution.pos)
import { solvedState, applyMove, applyMoves, invertMove } from './cube-model.js';

export class CubeSession {
  constructor() {
    this.reset();
  }

  reset() {
    this.state = solvedState();
    this.scramble = [];   // applied by 'scramble' turns after startScramble
    this.history = [];    // moves since the scramble, apart from a solution still showing
    this.solution = null; // { moves, seconds, base, pos, playing }, see showSolution
  }

  // Reset, then expect `moves` to arrive as 'scramble' turns.
  startScramble(moves) {
    this.reset();
    this.scramble = moves.slice();
  }

  // Record a turn that has finished animating. kind is 'user', 'undo',
  // 'scramble', 'solve' (a step forward through the solution) or
  // 'solve-back' (a step back). A solution step can still be animating when
  // the solution is put away; it then counts as an ordinary move, matching
  // what dropSolution did with the steps before it.
  applyTurn(move, kind) {
    this.state = applyMove(this.state, move);
    const s = this.solution;
    if (kind === 'user') this.history.push(move);
    else if (kind === 'undo') this.history.pop();
    else if (kind === 'solve') {
      if (s) s.pos++;
      else this.history.push(move);
    } else if (kind === 'solve-back') {
      if (s) s.pos--;
      else this.history.pop();
    }
  }

  // Show a solution for the current state. It starts at its first move, playing.
  showSolution(moves, seconds) {
    this.solution = { moves: moves.slice(), seconds, base: this.state.slice(), pos: 0, playing: true };
  }

  // Put the solution away. The moves already made from it join the history,
  // so the record still explains the cube and Undo can take them back.
  dropSolution() {
    const s = this.solution;
    if (!s) return;
    this.history.push(...s.moves.slice(0, s.pos));
    this.solution = null;
  }

  // The turn one step forward or back through the solution, or null at that end.
  nextSolutionTurn() {
    const s = this.solution;
    return s && s.pos < s.moves.length ? { move: s.moves[s.pos], kind: 'solve' } : null;
  }

  prevSolutionTurn() {
    const s = this.solution;
    return s && s.pos > 0 ? { move: invertMove(s.moves[s.pos - 1]), kind: 'solve-back' } : null;
  }

  // What Undo does: step back through a solution part way played, otherwise
  // take back the last move. null when there is nothing to undo.
  undoTurn() {
    const back = this.prevSolutionTurn();
    if (back) return back;
    if (this.history.length === 0) return null;
    return { move: invertMove(this.history[this.history.length - 1]), kind: 'undo' };
  }

  // Go straight to the state after the first k solution moves, paused.
  jumpTo(k) {
    const s = this.solution;
    if (!s) return;
    s.playing = false;
    s.pos = Math.max(0, Math.min(k, s.moves.length));
    this.state = applyMoves(s.base, s.moves.slice(0, s.pos));
  }

  // The moves from a solved cube to the current state, according to the record.
  movesFromSolved() {
    const s = this.solution;
    return [...this.scramble, ...this.history, ...(s ? s.moves.slice(0, s.pos) : [])];
  }
}

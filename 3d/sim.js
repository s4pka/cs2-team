// Grenade flight against CS2 collision triangles.
// A line-by-line port of NC1107/cs2-smoke-solver, commit 7ebba5b (2026-09-23):
//   src/Sim/GrenadeTrajectory.cs  ThrowConstants, Bounce, ForwardFromAngles, DeriveInitial, SimulateExactRaw
//   src/Sim/TriangleCollider*.cs  uniform grid (128 u cells), FirstHitHullIndexed, SweptBoxTriangle
//   src/Sim/CollisionMesh.cs      GrenadeSolidFilter (applied when the mesh file was built, see geo.py)
// Units are Hammer units, z up. Math runs in float64 here, float32 in the original.

export const K = {
  ThrowSpeed: 675, GravityScale: 0.40, Elasticity: 0.45, StopSpeed: 19.685, DampGateSpeed: 690,
  JumpVelocity: 273.6, CrouchJumpVelocity: 277.5, BouncesPerTick: 2, GlassPassFactor: 0.40, RunSpeed: 306,
  ReleaseRiseRight: 14.0, ReleaseRiseBoth: 20.0, ReleaseRiseLeft: 26.1, RightClickScale: 0.30, BothClickScale: 0.65,
};
export const STAND_EYE = 64.06, CROUCH_EYE = 46.04;
const TimeStep = 1 / 64, MaxFlightSeconds = 10, MaxVel = 3500, StopEpsilon = 0.1, BaseGravity = 800;
const FloorNormalZ = 0.7, GrenadeRadius = 2, BrokenPaneReach = 96, Substeps = 2;
const H = [GrenadeRadius, GrenadeRadius, GrenadeRadius];

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = a => Math.hypot(a[0], a[1], a[2]);
const norm = a => { const l = len(a); return [a[0] / l, a[1] / l, a[2] / l]; };
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const dist2 = (a, b) => { const d = sub(a, b); return dot(d, d); };

export class Collider {
  // verts: Float32Array xyz, idx: Uint32Array, breakable: Uint8Array per triangle (1 = glass)
  constructor(verts, idx, breakable, cell = 128) {
    this.v = verts; this.i = idx; this.br = breakable; this.cs = cell;
    const n = idx.length / 3, tb = new Float32Array(n * 6);
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let t = 0; t < n; t++) {
      for (let k = 0; k < 3; k++) {
        const a = verts[idx[t * 3] * 3 + k], b = verts[idx[t * 3 + 1] * 3 + k], c = verts[idx[t * 3 + 2] * 3 + k];
        tb[t * 6 + k] = Math.min(a, b, c); tb[t * 6 + 3 + k] = Math.max(a, b, c);
        lo[k] = Math.min(lo[k], tb[t * 6 + k]); hi[k] = Math.max(hi[k], tb[t * 6 + 3 + k]);
      }
    }
    this.tb = tb; this.o = lo;
    this.nx = Math.max(1, Math.ceil((hi[0] - lo[0]) / cell));
    this.ny = Math.max(1, Math.ceil((hi[1] - lo[1]) / cell));
    this.nz = Math.max(1, Math.ceil((hi[2] - lo[2]) / cell));
    const cc = this.nx * this.ny * this.nz, start = new Int32Array(cc + 1);
    const visit = (t, f) => {
      const [x0, y0, z0] = this.cellOf([tb[t * 6], tb[t * 6 + 1], tb[t * 6 + 2]]);
      const [x1, y1, z1] = this.cellOf([tb[t * 6 + 3], tb[t * 6 + 4], tb[t * 6 + 5]]);
      for (let z = Math.max(z0, 0); z <= Math.min(z1, this.nz - 1); z++)
        for (let y = Math.max(y0, 0); y <= Math.min(y1, this.ny - 1); y++)
          for (let x = Math.max(x0, 0); x <= Math.min(x1, this.nx - 1); x++) f((z * this.ny + y) * this.nx + x);
    };
    for (let t = 0; t < n; t++) visit(t, c => start[c + 1]++);
    for (let c = 0; c < cc; c++) start[c + 1] += start[c];
    const tris = new Int32Array(start[cc]), fill = new Int32Array(cc);
    for (let t = 0; t < n; t++) visit(t, c => { tris[start[c] + fill[c]++] = t; });
    this.start = start; this.tris = tris;
  }
  cellOf(p) { return [0, 1, 2].map(k => Math.floor((p[k] - this.o[k]) / this.cs)); }
  vert(j) { return [this.v[j * 3], this.v[j * 3 + 1], this.v[j * 3 + 2]]; }
  tri(t) { return [this.vert(this.i[t * 3]), this.vert(this.i[t * 3 + 1]), this.vert(this.i[t * 3 + 2])]; }
  isBreakable(t) { return this.br[t] === 1; }
  centroid(t) { const [a, b, c] = this.tri(t); return mul(add(add(a, b), c), 1 / 3); }

  // FirstHitHullIndexed: closest swept-box contact on from -> to, or null.
  hull(from, to, minNormalZ = -2, ignore = null) {
    const d = sub(to, from), lo = [], hi = [];
    for (let k = 0; k < 3; k++) { lo[k] = Math.min(from[k], to[k]) - H[k]; hi[k] = Math.max(from[k], to[k]) + H[k]; }
    const [x0, y0, z0] = this.cellOf(lo), [x1, y1, z1] = this.cellOf(hi);
    let best = null, seen = new Set();
    for (let z = Math.max(z0, 0); z <= Math.min(z1, this.nz - 1); z++)
      for (let y = Math.max(y0, 0); y <= Math.min(y1, this.ny - 1); y++)
        for (let x = Math.max(x0, 0); x <= Math.min(x1, this.nx - 1); x++) {
          const c = (z * this.ny + y) * this.nx + x;
          for (let i = this.start[c]; i < this.start[c + 1]; i++) {
            const t = this.tris[i];
            if (seen.has(t)) continue; seen.add(t);
            const o = t * 6, tb = this.tb;
            if (!(tb[o + 3] >= lo[0] && tb[o] <= hi[0] && tb[o + 4] >= lo[1] && tb[o + 1] <= hi[1] && tb[o + 5] >= lo[2] && tb[o + 2] <= hi[2])) continue;
            if (ignore && ignore(t)) continue;
            const h = this.sweep(from, d, t);
            if (h && (!best || h.T < best.T) && h.N[2] >= minNormalZ) best = { ...h, tri: t };
          }
        }
    return best && best.T <= 1 ? best : null;
  }

  // SweptBoxTriangle: separating-axis sweep of the +-2 u box.
  sweep(origin, dir, t) {
    const [A, B, C] = this.tri(t), a = sub(A, origin), b = sub(B, origin), c = sub(C, origin);
    const n = cross(sub(b, a), sub(c, a));
    if (dot(n, n) < 1e-12) return null;
    let tEnter = 0, tExit = 1, enter = n;
    const axis = l => {
      if (dot(l, l) < 1e-10) return true;
      const s0 = dot(a, l), s1 = dot(b, l), s2 = dot(c, l);
      const m = Math.min(s0, s1, s2), M = Math.max(s0, s1, s2);
      const r = H[0] * Math.abs(l[0]) + H[1] * Math.abs(l[1]) + H[2] * Math.abs(l[2]);
      const w = dot(dir, l);
      if (Math.abs(w) < 1e-9) return m <= r && M >= -r;
      const tA = (m - r) / w, tB = (M + r) / w;
      const [axLo, axHi] = w > 0 ? [tA, tB] : [tB, tA];
      if (axLo > tEnter) { tEnter = axLo; enter = l; }
      tExit = Math.min(tExit, axHi);
      return tEnter <= tExit;
    };
    const X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1];
    if (!axis(X) || !axis(Y) || !axis(Z) || !axis(n)) return null;
    for (const e of [sub(b, a), sub(c, b), sub(a, c)])
      if (!axis(cross(X, e)) || !axis(cross(Y, e)) || !axis(cross(Z, e))) return null;
    if (tEnter <= 1e-6) {
      let n0 = norm(n);
      if (dot(a, n0) > 0) n0 = mul(n0, -1);
      return dot(norm(dir), n0) < -0.01 ? { T: 0, N: n0, face: true } : null;
    }
    let N = norm(enter);
    if (dot(N, dir) > 0) N = mul(N, -1);
    return { T: tEnter, N, face: enter === n };
  }
}

const speedScale = s => s >= 0.99 ? 1 : s >= 0.49 ? K.BothClickScale : K.RightClickScale;
const releaseRise = s => s >= 0.99 ? K.ReleaseRiseLeft : s >= 0.49 ? K.ReleaseRiseBoth : K.ReleaseRiseRight;

export function forward(pitchDeg, yawDeg) {
  const p = pitchDeg * Math.PI / 180, y = yawDeg * Math.PI / 180;
  return [Math.cos(p) * Math.cos(y), Math.cos(p) * Math.sin(y), -Math.sin(p)];
}

// type: 'stand' | 'crouch' | 'jump' | 'crouchjump' | 'runjump'; strength 1 = left, 0.5 = both, 0 = right
export function deriveInitial(eye, yawDeg, pitchDeg, type, strength = 1, runYawOffsetDeg = 0) {
  const ep = pitchDeg - (90 - Math.abs(pitchDeg)) / 90 * 10;
  const f = forward(ep, yawDeg);
  let vel = mul(f, K.ThrowSpeed * speedScale(strength));
  const rel = add(eye, mul(f, 16));
  if (type === 'jump' || type === 'crouchjump' || type === 'runjump') {
    vel[2] += type === 'crouchjump' ? K.CrouchJumpVelocity : K.JumpVelocity;
    rel[2] += releaseRise(strength);
  }
  if (type === 'runjump') {
    const ry = (yawDeg + runYawOffsetDeg) * Math.PI / 180;
    vel = add(vel, [Math.cos(ry) * K.RunSpeed, Math.sin(ry) * K.RunSpeed, 0]);
  }
  return [rel, vel];
}

const clampV = v => v.map(x => Math.max(-MaxVel, Math.min(MaxVel, x)));
const snap = v => v.map(x => Math.abs(x) < StopEpsilon ? 0 : x);
function bounce(w, n, gate) {
  const speed = len(w), wn = dot(w, n);
  const reflected = snap(sub(w, mul(n, 2 * wn)));
  const u = speed > 1e-6 ? Math.abs(wn) / speed : 0;
  const damp = gate > K.DampGateSpeed && u > 0.5 && n[2] > FloorNormalZ ? 1.5 - u : 1;
  return mul(reflected, K.Elasticity * damp);
}

// SimulateExactRaw. Returns rest point, path (one point per substep), bounce points, flight time, lost flag.
export function simulate(col, pos, vel) {
  const gStep = BaseGravity * K.GravityScale * TimeStep;
  let time = 0, bounces = 0, broken = null, ignore = null;
  const path = [pos], hits = [];
  while (time < MaxFlightSeconds) {
    const dt = TimeStep / Substeps, g = gStep / Substeps;
    for (let step = 0; step < Substeps; step++) {
      vel = clampV(vel);
      const vz = vel[2]; vel = [vel[0], vel[1], vel[2] - g];
      const next = add(pos, mul([vel[0], vel[1], (vz + vel[2]) * 0.5], dt));
      const hit = col.hull(pos, next, -2, ignore);
      if (!hit) { pos = next; path.push(pos); continue; }
      const contact = lerp(pos, next, Math.max(0, hit.T - 1e-3));
      pos = contact; bounces++; path.push(pos);
      if (K.GlassPassFactor > 0 && col.isBreakable(hit.tri)) {
        (broken ??= []).push(contact);
        const panes = broken;
        ignore = t => col.isBreakable(t) && panes.some(b => dist2(b, col.centroid(t)) < BrokenPaneReach ** 2);
        vel = mul(vel, K.GlassPassFactor);
        hits.push({ p: contact, glass: true });
        const through = add(pos, mul(vel, (1 - hit.T) * dt));
        const behind = col.hull(pos, through, -2, ignore);
        pos = behind ? lerp(pos, through, Math.max(0, behind.T - 1e-3)) : through;
        path.push(pos);
        continue;
      }
      const w = vel;
      const gate = len([w[0], w[1], w[2] - g * (Substeps - 1 - step)]);
      const vAfter = bounce(w, hit.N, gate);
      hits.push({ p: contact, n: hit.N });
      if (len(vAfter) < K.StopSpeed && (hit.N[2] > FloorNormalZ || col.hull(pos, add(pos, [0, 0, -2]), FloorNormalZ, null))) {
        return { rest: pos, path, hits, bounces, time: time + TimeStep, lost: false };
      }
      vel = vAfter;
      let remainder = 1 - hit.T, next2 = add(pos, mul(vAfter, remainder * dt));
      for (let s = 1; ; s++) {
        const h2 = col.hull(pos, next2, -2, ignore);
        if (!h2) { pos = next2; break; }
        pos = lerp(pos, next2, Math.max(0, h2.T - 1e-3));
        remainder *= 1 - h2.T;
        if (s >= K.BouncesPerTick || remainder <= 1e-4) break;
        bounces++;
        const w2 = vel;
        vel = bounce(w2, h2.N, len([w2[0], w2[1], w2[2] - g * (Substeps - 1 - step)]));
        hits.push({ p: pos, n: h2.N });
        next2 = add(pos, mul(vel, remainder * dt));
      }
      path.push(pos);
    }
    time += TimeStep;
  }
  return { rest: pos, path, hits, bounces, time, lost: true };
}

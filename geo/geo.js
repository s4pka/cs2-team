/*
 * geo.js - map geometry for the CS2 team site's tactics board: line of sight, view polygons, smoke shapes, who-sees-a-spot,
 * walking paths. Plain script, no build step. Browser: window.Geo. Node: module.exports = { Geo, GeoCore }.
 *
 * Data: files made by build_geo.py (data/<map>.mesh.gz, data/<map>.nav.gz, data/maps.json) from pnxenopoulos/awpy-data
 * release 2000942 (collision mesh "AWMH", nav version 36).
 *
 * Ported or based on (pnxenopoulos/awpy, main at c3adcc6e417255e14f748952978c553ddbe1a1b3, read 2026-11-09):
 *  - crates/awpy/src/geometry.rs: VisibilityMesh. A bounding-volume hierarchy built by midpoint split of the triangle
 *    centroids on the longest axis, leaf size 4; a segment is blocked when Moeller-Trumbore (two-sided) hits a triangle
 *    with the segment fraction in (1e-4, 1 - 1e-4). Same constants here (LEAF_SIZE, SEG_EPS, PARALLEL_EPS).
 *  - crates/awpy/src/map_control.rs and crates/awpy-python/python/awpy/map_control.py: Params defaults eye_height 64,
 *    crouch_eye_height 46, target_height 46; smokes are spheres, radius 144 (smoke_radius), centre 60 above the
 *    landing point (smoke_height); a sight line is cut when the segment passes within the radius of the centre
 *    (segment_hits_sphere).
 *  - crates/awpy/src/nav.rs: Nav::find_area (point in polygon in xy, the area whose height is nearest wins),
 *    find_path (Dijkstra over connections, edge cost = 3D distance between area centroids; here A* with the same cost).
 * Own additions (not in awpy): terrain-following view rays, flood-filled smoke footprint, path smoothing, ladder edges
 * (made in build_geo.py), who-sees-this-spot sampling, the radar-pixel conversions.
 *
 * Coordinates in the API: the board's radar-pixel space (the site's cropped 1024 px radar):
 *   world x = posX + (px + cropX) * scale,   world y = posY - (py + cropY) * scale     (state.md, "Grenade map")
 * A point is [px, py] or [px, py, floorZ] (world height of the floor under the feet, when known, e.g. from the
 * lineup's setpos). Angles are the board's: degrees, atan2(dy, dx) in pixels (0 = right, clockwise on screen).
 * Every method returns a Promise: the work runs in a Web Worker.
 */

/* ---------- the core: runs in the worker (and in Node for tests). Must not use anything outside itself. ---------- */
function GeoCore() {
  'use strict';
  const LEAF_SIZE = 4, SEG_EPS = 1e-4, PARALLEL_EPS = 1e-8;
  const EYE = 64, CROUCH = 46, TARGET = 46;                 // awpy Params defaults
  const SMOKE_R = 144, SMOKE_H = 60;                        // awpy smoke_radius, smoke_height
  const SPEED = { knife: 250, rifle: 215 };                 // units per second: cs2.gs/weapons/knife (250), cs2.gs/weapons/ak-47 (215), counterstrikewiki.com/Movement_Mechanics (read 2026-11-09)
  let M = null;                                             // map frame: posX, posY, scale, cx, cy, lowerBelowZ

  /* ----- mesh + BVH ----- */
  let nT = 0, TRI = null, ORD = null, NB = null, NS = null, NC = null, NL = null, NR = null, nn = 0;
  const stack = new Int32Array(256);

  function loadMesh(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x31534d47) throw new Error('mesh: bad magic');          // 'GMS1'
    const nV = dv.getUint32(4, true); nT = dv.getUint32(8, true);
    const lo = [dv.getFloat32(12, true), dv.getFloat32(16, true), dv.getFloat32(20, true)];
    const st = [dv.getFloat32(24, true), dv.getFloat32(28, true), dv.getFloat32(32, true)];
    let off = 36;
    const V = new Float32Array(nV * 3);
    for (let ax = 0; ax < 3; ax++) {
      const d = new Int16Array(buf.slice(off, off + nV * 2)); off += nV * 2;
      let q = 0; const l = lo[ax], s = st[ax];
      for (let i = 0; i < nV; i++) { q = (q + d[i]) & 0xffff; V[i * 3 + ax] = l + q * s; }
    }
    off = (off + 3) & ~3;
    const rel = new Int32Array(buf.slice(off, off + nT * 12));
    const T = new Int32Array(nT * 3);
    let pm = -1;
    for (let i = 0; i < nT; i++) {
      const a = pm + 1 - rel[3 * i], b = pm + 1 - rel[3 * i + 1], c = pm + 1 - rel[3 * i + 2];
      T[3 * i] = a; T[3 * i + 1] = b; T[3 * i + 2] = c;
      if (a > pm) pm = a; if (b > pm) pm = b; if (c > pm) pm = c;
    }
    buildBvh(V, T);
  }

  function buildBvh(V, T) {
    const tb = new Float32Array(nT * 6), tc = new Float32Array(nT * 3);
    for (let i = 0; i < nT; i++) {
      const a = T[3 * i] * 3, b = T[3 * i + 1] * 3, c = T[3 * i + 2] * 3;
      for (let k = 0; k < 3; k++) {
        const x = V[a + k], y = V[b + k], z = V[c + k];
        const mn = Math.min(x, y, z), mx = Math.max(x, y, z);
        tb[i * 6 + k] = mn; tb[i * 6 + 3 + k] = mx; tc[i * 3 + k] = 0.5 * (mn + mx);
      }
    }
    const order = new Uint32Array(nT);
    for (let i = 0; i < nT; i++) order[i] = i;
    let cap = nT + 16;
    NB = new Float32Array(cap * 6); NS = new Int32Array(cap); NC = new Int32Array(cap); NL = new Int32Array(cap); NR = new Int32Array(cap); nn = 0;
    function grow() {
      const c2 = cap * 2, nb = new Float32Array(c2 * 6); nb.set(NB); NB = nb;
      for (const nm of ['NS', 'NC', 'NL', 'NR']) { const n = new Int32Array(c2); n.set(nm === 'NS' ? NS : nm === 'NC' ? NC : nm === 'NL' ? NL : NR); if (nm === 'NS') NS = n; else if (nm === 'NC') NC = n; else if (nm === 'NL') NL = n; else NR = n; }
      cap = c2;
    }
    function build(start, end) {
      if (nn >= cap) grow();
      const idx = nn++;
      let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
      for (let k = start; k < end; k++) {
        const t = order[k] * 6;
        if (tb[t] < x0) x0 = tb[t]; if (tb[t + 1] < y0) y0 = tb[t + 1]; if (tb[t + 2] < z0) z0 = tb[t + 2];
        if (tb[t + 3] > x1) x1 = tb[t + 3]; if (tb[t + 4] > y1) y1 = tb[t + 4]; if (tb[t + 5] > z1) z1 = tb[t + 5];
      }
      NB[idx * 6] = x0; NB[idx * 6 + 1] = y0; NB[idx * 6 + 2] = z0; NB[idx * 6 + 3] = x1; NB[idx * 6 + 4] = y1; NB[idx * 6 + 5] = z1;
      NS[idx] = start; NC[idx] = end - start;
      if (end - start <= LEAF_SIZE) return idx;
      let cx0 = Infinity, cy0 = Infinity, cz0 = Infinity, cx1 = -Infinity, cy1 = -Infinity, cz1 = -Infinity;
      for (let k = start; k < end; k++) {
        const t = order[k] * 3, x = tc[t], y = tc[t + 1], z = tc[t + 2];
        if (x < cx0) cx0 = x; if (x > cx1) cx1 = x; if (y < cy0) cy0 = y; if (y > cy1) cy1 = y; if (z < cz0) cz0 = z; if (z > cz1) cz1 = z;
      }
      const ex = cx1 - cx0, ey = cy1 - cy0, ez = cz1 - cz0;
      const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
      const split = 0.5 * ((axis === 0 ? cx0 : axis === 1 ? cy0 : cz0) + (axis === 0 ? cx1 : axis === 1 ? cy1 : cz1));
      let i = start;
      for (let j = start; j < end; j++) {
        if (tc[order[j] * 3 + axis] < split) { const t = order[i]; order[i] = order[j]; order[j] = t; i++; }
      }
      let mid = i;
      if (mid === start || mid === end) {                         // degenerate split: median
        const sub = Array.from(order.subarray(start, end)).sort((p, q) => tc[p * 3 + axis] - tc[q * 3 + axis]);
        for (let k = 0; k < sub.length; k++) order[start + k] = sub[k];
        mid = start + ((end - start) >> 1);
      }
      const l = build(start, mid), r = build(mid, end);
      NL[idx] = l; NR[idx] = r; NC[idx] = 0;
      return idx;
    }
    if (nT > 0) build(0, nT);
    ORD = order;
    // triangles in leaf order: v0, e1 = v1 - v0, e2 = v2 - v0
    TRI = new Float32Array(nT * 9);
    for (let k = 0; k < nT; k++) {
      const t = order[k], a = T[3 * t] * 3, b = T[3 * t + 1] * 3, c = T[3 * t + 2] * 3, o = k * 9;
      TRI[o] = V[a]; TRI[o + 1] = V[a + 1]; TRI[o + 2] = V[a + 2];
      TRI[o + 3] = V[b] - V[a]; TRI[o + 4] = V[b + 1] - V[a + 1]; TRI[o + 5] = V[b + 2] - V[a + 2];
      TRI[o + 6] = V[c] - V[a]; TRI[o + 7] = V[c + 1] - V[a + 1]; TRI[o + 8] = V[c + 2] - V[a + 2];
    }
  }

  // Is the segment a-b blocked by the mesh? (awpy VisibilityMesh::is_occluded)
  function occluded(ax, ay, az, bx, by, bz) {
    if (nn === 0) return false;
    const dx = bx - ax, dy = by - ay, dz = bz - az;
    if (dx === 0 && dy === 0 && dz === 0) return false;
    const ix = 1 / dx, iy = 1 / dy, iz = 1 / dz;
    let sp = 0; stack[sp++] = 0;
    while (sp > 0) {
      const ni = stack[--sp], o = ni * 6;
      let tmin = 0, tmax = 1, ta, tb2;
      ta = (NB[o] - ax) * ix; tb2 = (NB[o + 3] - ax) * ix; if (ix < 0) { const s = ta; ta = tb2; tb2 = s; }
      if (ta > tmin) tmin = ta; if (tb2 < tmax) tmax = tb2; if (tmax < tmin) continue;
      ta = (NB[o + 1] - ay) * iy; tb2 = (NB[o + 4] - ay) * iy; if (iy < 0) { const s = ta; ta = tb2; tb2 = s; }
      if (ta > tmin) tmin = ta; if (tb2 < tmax) tmax = tb2; if (tmax < tmin) continue;
      ta = (NB[o + 2] - az) * iz; tb2 = (NB[o + 5] - az) * iz; if (iz < 0) { const s = ta; ta = tb2; tb2 = s; }
      if (ta > tmin) tmin = ta; if (tb2 < tmax) tmax = tb2; if (tmax < tmin) continue;
      const cnt = NC[ni];
      if (cnt > 0) {
        for (let k = NS[ni], e = k + cnt; k < e; k++) {
          const q = k * 9;
          const e1x = TRI[q + 3], e1y = TRI[q + 4], e1z = TRI[q + 5], e2x = TRI[q + 6], e2y = TRI[q + 7], e2z = TRI[q + 8];
          const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
          const det = e1x * px + e1y * py + e1z * pz;
          if (det < PARALLEL_EPS && det > -PARALLEL_EPS) continue;
          const inv = 1 / det, sx = ax - TRI[q], sy = ay - TRI[q + 1], sz = az - TRI[q + 2];
          const u = (sx * px + sy * py + sz * pz) * inv;
          if (u < 0 || u > 1) continue;
          const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
          const v = (dx * qx + dy * qy + dz * qz) * inv;
          if (v < 0 || u + v > 1) continue;
          const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
          if (t > SEG_EPS && t < 1 - SEG_EPS) return true;
        }
      } else { stack[sp++] = NL[ni]; stack[sp++] = NR[ni]; }
    }
    return false;
  }

  // Distance along the unit direction (dx,dy,dz) from o to the first triangle, or -1 if none within tmax.
  function closest(ox, oy, oz, dx, dy, dz, tmax0) {
    if (nn === 0) return -1;
    const ix = 1 / dx, iy = 1 / dy, iz = 1 / dz;
    let best = tmax0, hit = false;
    let sp = 0; stack[sp++] = 0;
    while (sp > 0) {
      const ni = stack[--sp], o = ni * 6;
      let tmin = 0, tmax = best, ta, tb2;
      ta = (NB[o] - ox) * ix; tb2 = (NB[o + 3] - ox) * ix; if (ix < 0) { const s = ta; ta = tb2; tb2 = s; }
      if (ta > tmin) tmin = ta; if (tb2 < tmax) tmax = tb2; if (tmax < tmin) continue;
      ta = (NB[o + 1] - oy) * iy; tb2 = (NB[o + 4] - oy) * iy; if (iy < 0) { const s = ta; ta = tb2; tb2 = s; }
      if (ta > tmin) tmin = ta; if (tb2 < tmax) tmax = tb2; if (tmax < tmin) continue;
      ta = (NB[o + 2] - oz) * iz; tb2 = (NB[o + 5] - oz) * iz; if (iz < 0) { const s = ta; ta = tb2; tb2 = s; }
      if (ta > tmin) tmin = ta; if (tb2 < tmax) tmax = tb2; if (tmax < tmin) continue;
      const cnt = NC[ni];
      if (cnt > 0) {
        for (let k = NS[ni], e = k + cnt; k < e; k++) {
          const q = k * 9;
          const e1x = TRI[q + 3], e1y = TRI[q + 4], e1z = TRI[q + 5], e2x = TRI[q + 6], e2y = TRI[q + 7], e2z = TRI[q + 8];
          const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
          const det = e1x * px + e1y * py + e1z * pz;
          if (det < PARALLEL_EPS && det > -PARALLEL_EPS) continue;
          const inv = 1 / det, sx = ox - TRI[q], sy = oy - TRI[q + 1], sz = oz - TRI[q + 2];
          const u = (sx * px + sy * py + sz * pz) * inv;
          if (u < 0 || u > 1) continue;
          const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
          const v = (dx * qx + dy * qy + dz * qz) * inv;
          if (v < 0 || u + v > 1) continue;
          const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
          if (t > 0.05 && t < best) { best = t; hit = true; }
        }
      } else { stack[sp++] = NL[ni]; stack[sp++] = NR[ni]; }
    }
    return hit ? best : -1;
  }

  /* ----- nav ----- */
  let nA = 0, CX = null, CY = null, CZ = null, PS = null, PI = null, KS = null, KI = null;
  let ACX = null, ACY = null, ACZ = null, ABB = null, APL = null, AZR = null;
  const G = 128;
  let gx0 = 0, gy0 = 0, gw = 0, gh = 0, gStart = null, gItems = null;

  function loadNav(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x31564e47) throw new Error('nav: bad magic');          // 'GNV1'
    const nC = dv.getUint32(4, true); nA = dv.getUint32(8, true);
    const nP = dv.getUint32(12, true), nK = dv.getUint32(16, true);
    const lo = [dv.getFloat32(20, true), dv.getFloat32(24, true), dv.getFloat32(28, true)];
    const st = [dv.getFloat32(32, true), dv.getFloat32(36, true), dv.getFloat32(40, true)];
    let off = 48;
    const p4 = n => (n + 3) & ~3;
    const cq = new Uint16Array(buf.slice(off, off + nC * 6)); off += p4(nC * 6);
    CX = new Float32Array(nC); CY = new Float32Array(nC); CZ = new Float32Array(nC);
    for (let i = 0; i < nC; i++) { CX[i] = lo[0] + cq[i] * st[0]; CY[i] = lo[1] + cq[nC + i] * st[1]; CZ[i] = lo[2] + cq[2 * nC + i] * st[2]; }
    const plen = new Uint8Array(buf, off, nA); off += p4(nA);
    PI = new Uint16Array(buf.slice(off, off + nP * 2)); off += p4(nP * 2);
    const klen = new Uint8Array(buf, off, nA); off += p4(nA);
    KI = new Uint16Array(buf.slice(off, off + nK * 2));
    PS = new Int32Array(nA + 1); KS = new Int32Array(nA + 1);
    for (let i = 0; i < nA; i++) { PS[i + 1] = PS[i] + plen[i]; KS[i + 1] = KS[i] + klen[i]; }
    ACX = new Float32Array(nA); ACY = new Float32Array(nA); ACZ = new Float32Array(nA); ABB = new Float32Array(nA * 4); APL = new Float32Array(nA * 4); AZR = new Float32Array(nA * 2);
    let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
    for (let i = 0; i < nA; i++) {
      const s = PS[i], e = PS[i + 1], n = e - s;
      let sx = 0, sy = 0, sz = 0, x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, z0 = Infinity, z1 = -Infinity;
      for (let k = s; k < e; k++) {
        const c = PI[k], x = CX[c], y = CY[c], z = CZ[c];
        sx += x; sy += y; sz += z;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (z < z0) z0 = z; if (z > z1) z1 = z;
      }
      ACX[i] = sx / n; ACY[i] = sy / n; ACZ[i] = sz / n;
      ABB[i * 4] = x0; ABB[i * 4 + 1] = y0; ABB[i * 4 + 2] = x1; ABB[i * 4 + 3] = y1; AZR[i * 2] = z0; AZR[i * 2 + 1] = z1;
      // plane z = z0p + a (x - x0p) + b (y - y0p) through the first three corners (when not vertical)
      const c0 = PI[s], c1 = PI[s + 1], c2 = PI[s + 2];
      const ux = CX[c1] - CX[c0], uy = CY[c1] - CY[c0], uz = CZ[c1] - CZ[c0], vx = CX[c2] - CX[c0], vy = CY[c2] - CY[c0], vz = CZ[c2] - CZ[c0];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      if (Math.abs(nz) > 1e-3) { APL[i * 4] = -nx / nz; APL[i * 4 + 1] = -ny / nz; APL[i * 4 + 2] = CX[c0]; APL[i * 4 + 3] = CY[c0]; } else { APL[i * 4] = 0; APL[i * 4 + 1] = 0; APL[i * 4 + 2] = 0; APL[i * 4 + 3] = 0; }
      if (x0 < bx0) bx0 = x0; if (y0 < by0) by0 = y0; if (x1 > bx1) bx1 = x1; if (y1 > by1) by1 = y1;
    }
    // uniform grid of areas by bounding box
    gx0 = Math.floor(bx0 / G) * G; gy0 = Math.floor(by0 / G) * G; gw = Math.ceil((bx1 - gx0) / G) + 1; gh = Math.ceil((by1 - gy0) / G) + 1;
    const cnt = new Int32Array(gw * gh + 1);
    for (let pass = 0; pass < 2; pass++) {
      if (pass === 1) { for (let i = 0; i < gw * gh; i++) cnt[i + 1] += cnt[i]; gItems = new Int32Array(cnt[gw * gh]); gStart = cnt.slice(); }
      const fill = pass === 1 ? gStart.slice(0, gw * gh) : null;
      for (let a = 0; a < nA; a++) {
        const i0 = Math.floor((ABB[a * 4] - gx0) / G), i1 = Math.floor((ABB[a * 4 + 2] - gx0) / G), j0 = Math.floor((ABB[a * 4 + 1] - gy0) / G), j1 = Math.floor((ABB[a * 4 + 3] - gy0) / G);
        for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
          if (pass === 0) cnt[j * gw + i + 1]++; else gItems[fill[j * gw + i]++] = a;
        }
      }
    }
  }

  function inPoly(a, x, y) {                                  // awpy NavArea::contains_xy (even-odd)
    const s = PS[a], e = PS[a + 1];
    let inside = false, j = e - 1;
    for (let i = s; i < e; i++) {
      const ci = PI[i], cj = PI[j], xi = CX[ci], yi = CY[ci], xj = CX[cj], yj = CY[cj];
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
      j = i;
    }
    return inside;
  }
  function distPoly(a, x, y) {
    if (inPoly(a, x, y)) return 0;
    const s = PS[a], e = PS[a + 1];
    let best = Infinity, j = e - 1;
    for (let i = s; i < e; i++) {
      const ci = PI[i], cj = PI[j], x1 = CX[cj], y1 = CY[cj], dx = CX[ci] - x1, dy = CY[ci] - y1;
      const l2 = dx * dx + dy * dy;
      let t = l2 > 0 ? ((x - x1) * dx + (y - y1) * dy) / l2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
      const d = Math.hypot(x - (x1 + t * dx), y - (y1 + t * dy));
      if (d < best) best = d;
      j = i;
    }
    return best;
  }
  function zOn(a, x, y) {
    let z = ACZ[a];
    if (APL[a * 4] !== 0 || APL[a * 4 + 1] !== 0) z = CZ[PI[PS[a]]] + APL[a * 4] * (x - APL[a * 4 + 2]) + APL[a * 4 + 1] * (y - APL[a * 4 + 3]);
    const lo = AZR[a * 2], hi = AZR[a * 2 + 1];
    return z < lo ? lo : z > hi ? hi : z;
  }
  const cand = [];
  function areasAt(x, y) {                                    // indices of areas containing (x,y); fills `cand`
    cand.length = 0;
    const i = Math.floor((x - gx0) / G), j = Math.floor((y - gy0) / G);
    if (i < 0 || j < 0 || i >= gw || j >= gh) return cand;
    const c = j * gw + i;
    for (let k = gStart[c]; k < gStart[c + 1]; k++) { const a = gItems[k]; if (inPoly(a, x, y)) cand.push(a); }
    return cand;
  }
  function levelOf(z) { return M && M.lowerBelowZ != null ? (z < M.lowerBelowZ ? 'lower' : 'upper') : null; }
  // Floor under (x,y). hint: number (a world floor height: the nearest floor wins), 'upper' / 'lower' (Nuke), or null.
  // Several floors with no hint: the lowest one on the default level (see README).
  function floorAt(x, y, hint) {
    const c = areasAt(x, y);
    if (c.length) {
      let best = -1, bd = Infinity;
      for (const a of c) {
        const z = zOn(a, x, y);
        let d;
        if (typeof hint === 'number') d = Math.abs(z - hint);
        else if (hint === 'upper' || hint === 'lower') d = levelOf(z) === hint ? z : 1e6 + z;
        else d = (levelOf(z) === 'lower' ? 1e6 : 0) + z;
        if (d < bd) { bd = d; best = a; }
      }
      return { area: best, z: zOn(best, x, y), snapped: false };
    }
    // snap to the nearest area within 40 units
    const R = 40;
    let best = -1, bd = R;
    const i0 = Math.floor((x - R - gx0) / G), i1 = Math.floor((x + R - gx0) / G), j0 = Math.floor((y - R - gy0) / G), j1 = Math.floor((y + R - gy0) / G);
    for (let j = Math.max(0, j0); j <= Math.min(gh - 1, j1); j++) for (let i = Math.max(0, i0); i <= Math.min(gw - 1, i1); i++) {
      const cc = j * gw + i;
      for (let k = gStart[cc]; k < gStart[cc + 1]; k++) {
        const a = gItems[k], d = distPoly(a, x, y);
        const z = ACZ[a];
        const pen = typeof hint === 'number' ? Math.abs(z - hint) * 0.01 : 0;
        if (d + pen < bd) { bd = d + pen; best = a; }
      }
    }
    return best >= 0 ? { area: best, z: zOn(best, x, y), snapped: true } : null;
  }
  // nearest floor height to a reference height within a tolerance (used while marching)
  function floorNear(x, y, zref, tol) {
    const c = areasAt(x, y);
    let bz = null, bd = tol;
    for (const a of c) { const z = zOn(a, x, y), d = Math.abs(z - zref); if (d <= bd) { bd = d; bz = z; } }
    return bz;
  }

  /* ----- frame ----- */
  const w2x = px => M.posX + (px + M.cx) * M.scale;
  const w2y = py => M.posY - (py + M.cy) * M.scale;
  const x2px = x => (x - M.posX) / M.scale - M.cx;
  const y2py = y => (M.posY - y) / M.scale - M.cy;
  function wpt(p, hint) {
    const x = w2x(p[0]), y = w2y(p[1]);
    let z = p[2], area = -1;
    const f = floorAt(x, y, typeof z === 'number' ? z : hint);
    if (typeof z !== 'number') { if (!f) return null; z = f.z; }
    if (f) area = f.area;
    return { x, y, z, area };
  }

  /* ----- smokes ----- */
  const smokes = [];
  let smokeId = 1;
  function inSmoke(s, x, y, z) {
    const dx = x - s.x, dy = y - s.y, dz = z - s.cz;
    if (dx * dx + dy * dy + dz * dz > s.R * s.R) return false;
    const i = Math.round(dx / s.cell) + s.H, j = Math.round(dy / s.cell) + s.H;
    return i >= 0 && j >= 0 && i < s.N && j < s.N && s.bits[j * s.N + i] === 1;
  }
  function smokeBlocks(ax, ay, az, bx, by, bz) {
    if (smokes.length === 0) return false;
    const dx = bx - ax, dy = by - ay, dz = bz - az, len = Math.hypot(dx, dy, dz);
    if (len === 0) return false;
    for (const s of smokes) {
      // segment-sphere interval
      const fx = ax - s.x, fy = ay - s.y, fz = az - s.cz;
      const A = dx * dx + dy * dy + dz * dz, B = 2 * (fx * dx + fy * dy + fz * dz), C = fx * fx + fy * fy + fz * fz - s.R * s.R;
      const disc = B * B - 4 * A * C;
      if (disc < 0) continue;
      const sq = Math.sqrt(disc);
      let t0 = (-B - sq) / (2 * A), t1 = (-B + sq) / (2 * A);
      if (t1 < 0 || t0 > 1) continue;
      if (t0 < 0) t0 = 0; if (t1 > 1) t1 = 1;
      const n = Math.ceil((t1 - t0) * len / 4) + 1;
      for (let k = 0; k <= n; k++) {
        const t = t0 + (t1 - t0) * k / n;
        if (inSmoke(s, ax + dx * t, ay + dy * t, az + dz * t)) return true;
      }
    }
    return false;
  }
  function blocked(ax, ay, az, bx, by, bz, useSmokes) {
    return occluded(ax, ay, az, bx, by, bz) || (useSmokes && smokeBlocks(ax, ay, az, bx, by, bz));
  }

  // Flood-filled footprint of a smoke landing at world (x,y) on floor height zf.
  // Own approximation: awpy's sphere (radius 144, centre 60 above the landing) cut to what the gas can reach: a 4-neighbour
  // flood fill on an 8-unit grid; a step is taken only when the segment between the two cell points (floor + 46)
  // is clear of the collision mesh, so smoke does not pass walls but turns corners into corridors.
  function floodSmoke(x, y, zf, opts) {
    const R = opts.radius || SMOKE_R, cell = opts.cell || 8, H = Math.ceil(R / cell), N = 2 * H + 1;
    const cz = zf + (opts.height != null ? opts.height : SMOKE_H);
    const bits = new Uint8Array(N * N), refz = new Float32Array(N * N);
    const qi = new Int32Array(N * N), qj = new Int32Array(N * N);
    let qh = 0, qt = 0, count = 0;
    bits[H * N + H] = 1; refz[H * N + H] = zf; qi[qt] = H; qj[qt++] = H; count = 1;
    const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    while (qh < qt) {
      const i = qi[qh], j = qj[qh++], zr = refz[j * N + i];
      const px = x + (i - H) * cell, py = y + (j - H) * cell, pz = zr + TARGET;
      for (const [di, dj] of dirs) {
        const ni = i + di, nj = j + dj;
        if (ni < 0 || nj < 0 || ni >= N || nj >= N || bits[nj * N + ni]) continue;
        const nx = x + (ni - H) * cell, ny = y + (nj - H) * cell;
        const nz = floorNear(nx, ny, zr, 24);
        const zr2 = nz === null ? zr : nz;
        const ex = nx - x, ey = ny - y, ez = zr2 + TARGET - cz;
        if (ex * ex + ey * ey + ez * ez > R * R) continue;
        if (occluded(px, py, pz, nx, ny, zr2 + TARGET)) continue;
        bits[nj * N + ni] = 1; refz[nj * N + ni] = zr2; qi[qt] = ni; qj[qt++] = nj; count++;
      }
    }
    return { x, y, zf, cz, R, cell, H, N, bits, count };
  }
  // outer boundary rings of the cell bitmap, as world polygons
  function rings(s) {
    const { N, H, cell, bits, x, y } = s;
    const nx = N + 1, out = new Map();                        // lattice vertex -> list of [toVertex]
    const add = (a, b) => { if (!out.has(a)) out.set(a, []); out.get(a).push(b); };
    const V = (i, j) => j * nx + i;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      if (!bits[j * N + i]) continue;
      // counter-clockwise (in x right, y up): bottom edge left->right, right edge up, top edge right->left, left edge down
      if (j === 0 || !bits[(j - 1) * N + i]) add(V(i, j), V(i + 1, j));
      if (i === N - 1 || !bits[j * N + i + 1]) add(V(i + 1, j), V(i + 1, j + 1));
      if (j === N - 1 || !bits[(j + 1) * N + i]) add(V(i + 1, j + 1), V(i, j + 1));
      if (i === 0 || !bits[j * N + i - 1]) add(V(i, j + 1), V(i, j));
    }
    const res = [];
    for (const [start] of out) {
      while (out.get(start) && out.get(start).length) {
        const ring = []; let cur = start, prev = -1;
        for (;;) {
          const lst = out.get(cur);
          if (!lst || !lst.length) break;
          let pick = 0;
          if (lst.length > 1 && prev >= 0) {                  // diagonal touch: turn left (keep the region on the left)
            const dx = (cur % nx) - (prev % nx), dy = Math.floor(cur / nx) - Math.floor(prev / nx);
            let bestc = -Infinity;
            for (let k = 0; k < lst.length; k++) {
              const ex = (lst[k] % nx) - (cur % nx), ey = Math.floor(lst[k] / nx) - Math.floor(cur / nx);
              const cr = dx * ey - dy * ex;
              if (cr > bestc) { bestc = cr; pick = k; }
            }
          }
          const nxt = lst.splice(pick, 1)[0];
          ring.push(cur); prev = cur; cur = nxt;
          if (cur === start) break;
        }
        if (ring.length >= 3) res.push(ring);
      }
    }
    const toWorld = v => [x + ((v % nx) - H - 0.5) * cell, y + (Math.floor(v / nx) - H - 0.5) * cell];
    return res.map(r => r.map(toWorld));
  }
  function dp(pts, tol) {                                     // Douglas-Peucker on a closed ring
    if (pts.length < 8) return pts;
    const open = pts.concat([pts[0]]);
    const keep = new Uint8Array(open.length); keep[0] = keep[open.length - 1] = 1;
    let far = 0, fd = -1;                                     // a closed ring has equal ends: split at the point farthest from the start
    for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[0][0], pts[i][1] - pts[0][1]); if (d > fd) { fd = d; far = i; } }
    keep[far] = 1;
    const st = [[0, far], [far, open.length - 1]];
    while (st.length) {
      const [a, b] = st.pop();
      let md = 0, mi = -1;
      const ax = open[a][0], ay = open[a][1], bx = open[b][0], by = open[b][1], l = Math.hypot(bx - ax, by - ay) || 1;
      for (let i = a + 1; i < b; i++) { const d = Math.abs((by - ay) * open[i][0] - (bx - ax) * open[i][1] + bx * ay - by * ax) / l; if (d > md) { md = d; mi = i; } }
      if (md > tol && mi > 0) { keep[mi] = 1; st.push([a, mi], [mi, b]); }
    }
    return open.filter((_, i) => keep[i]).slice(0, -1);
  }
  const ringArea = r => { let s = 0; for (let i = 0; i < r.length; i++) { const a = r[i], b = r[(i + 1) % r.length]; s += a[0] * b[1] - b[0] * a[1]; } return s / 2; };

  function smokeShape(p, opts) {
    opts = opts || {};
    const w = wpt(p, opts.hint); if (!w) return null;
    const s = floodSmoke(w.x, w.y, w.z, opts);
    const rs = rings(s).map(r => dp(r, s.cell * 0.45)).map(r => r.map(q => [x2px(q[0]), y2py(q[1])]));
    // y flips between world and pixels, so areas change sign; outer ring = the largest by |area|
    rs.sort((a, b) => Math.abs(ringArea(b)) - Math.abs(ringArea(a)));
    return { poly: rs[0] || [], rings: rs, center: [x2px(w.x), y2py(w.y)], radiusPx: s.R / M.scale, z: w.z, cells: s.count, areaUnits2: s.count * s.cell * s.cell, _s: s };
  }
  function addSmoke(p, opts) {
    const sh = smokeShape(p, opts); if (!sh) return null;
    const s = sh._s; s.id = smokeId++; smokes.push(s);
    delete sh._s; sh.id = s.id; return sh;
  }
  function removeSmoke(id) { const i = smokes.findIndex(s => s.id === id); if (i >= 0) smokes.splice(i, 1); return i >= 0; }
  function clearSmokes() { smokes.length = 0; return true; }

  /* ----- queries ----- */
  function eyeH(o, which) { const k = which === 'b' ? 'heightB' : 'heightA'; return o[k] != null ? o[k] : (o.crouch ? CROUCH : EYE); }

  function visible(a, b, o) {
    o = o || {};
    const A = wpt(a, o.hintA != null ? o.hintA : o.hint), B = wpt(b, o.hintB != null ? o.hintB : o.hint);
    if (!A || !B) return null;
    return !blocked(A.x, A.y, A.z + eyeH(o, 'a'), B.x, B.y, B.z + eyeH(o, 'b'), o.smokes !== false);
  }

  function viewPolygon(p, angleDeg, fovDeg, rangePx, o) {
    o = o || {};
    const A = wpt(p, o.hint); if (!A) return null;
    const eh = eyeH(o, 'a'), ez = A.z + eh, maxD = Math.min(rangePx * M.scale, 8000);
    const useS = o.smokes !== false, terrain = o.terrain !== false, th = o.targetHeight != null ? o.targetHeight : TARGET, step = o.step || 64;
    function reach(thetaDeg) {                                // distance (world units) the player sees along this ray
      const t = thetaDeg * Math.PI / 180, ux = Math.cos(t), uy = -Math.sin(t);
      if (!terrain) {
        let d = closest(A.x, A.y, ez, ux, uy, 0, maxD); if (d < 0) d = maxD;
        if (useS && smokes.length) {                          // first smoke sample along the ray
          for (let q = 0; q < d; q += 4) if (inAnySmoke(A.x + ux * q, A.y + uy * q, ez)) { d = q; break; }
        }
        return d;
      }
      let lo = 0, fzPrev = A.z;
      const test = (d) => {
        const x = A.x + ux * d, y = A.y + uy * d;
        const nz = floorNear(x, y, fzPrev, 40);
        const fz = nz === null ? fzPrev : nz;
        return { b: blocked(A.x, A.y, ez, x, y, fz + th, useS), fz };
      };
      for (let d = step; ; d += step) {
        const dd = Math.min(d, maxD), r = test(dd);
        if (r.b) {
          let l = lo, h = dd;
          for (let k = 0; k < 6; k++) { const m = 0.5 * (l + h); if (test(m).b) h = m; else l = m; }
          return l;
        }
        lo = dd; fzPrev = r.fz;
        if (dd >= maxD) return maxD;
      }
    }
    const inAnySmoke = (x, y, z) => { for (const s of smokes) if (inSmoke(s, x, y, z)) return true; return false; };
    const full = fovDeg >= 359.99, stepDeg = o.stepDeg || 1, depth = o.refine != null ? o.refine : 3;
    const n0 = Math.max(2, Math.ceil(fovDeg / stepDeg) + (full ? 0 : 1)), a0 = angleDeg - fovDeg / 2, da = fovDeg / (full ? n0 : n0 - 1);
    const ang = [], rr = [];
    for (let i = 0; i < n0; i++) { ang.push(a0 + i * da); rr.push(reach(a0 + i * da)); }
    if (full) { ang.push(a0 + fovDeg); rr.push(rr[0]); }
    const outA = [], outR = [];
    function refine(a1, r1, a2, r2, dpth) {
      if (dpth > 0 && Math.abs(r1 - r2) > Math.max(16, 0.03 * Math.min(r1, r2))) {
        const am = 0.5 * (a1 + a2), rm = reach(am);
        refine(a1, r1, am, rm, dpth - 1); outA.push(am); outR.push(rm); refine(am, rm, a2, r2, dpth - 1);
      }
    }
    for (let i = 0; i < ang.length; i++) {
      outA.push(ang[i]); outR.push(rr[i]);
      if (i + 1 < ang.length) refine(ang[i], rr[i], ang[i + 1], rr[i + 1], depth);
    }
    const ex = x2px(A.x), ey = y2py(A.y), pts = [];
    if (!full) pts.push([ex, ey]);
    for (let i = 0; i < outA.length; i++) {
      const t = outA[i] * Math.PI / 180, r = outR[i] / M.scale;
      pts.push([ex + Math.cos(t) * r, ey + Math.sin(t) * r]);
    }
    return { poly: pts, rays: outA.length, z: A.z };
  }

  function seenFrom(p, o) {
    o = o || {};
    const T = wpt(p, o.hint); if (!T) return null;
    const useS = o.smokes !== false, th = o.targetHeight != null ? o.targetHeight : TARGET, eh = eyeH(o, 'a'), maxD = (o.rangePx || 1e9) * M.scale;
    const tz = T.z + th, res = [];
    for (let a = 0; a < nA; a++) {
      if (Math.hypot(ACX[a] - T.x, ACY[a] - T.y) > maxD) continue;
      const s = PS[a], n = PS[a + 1] - s, cx = ACX[a], cy = ACY[a];
      let seen = 0, total = 0;
      const sample = (x, y) => {
        total++;
        const z = zOn(a, x, y);
        if (!blocked(x, y, z + eh, T.x, T.y, tz, useS)) seen++;
      };
      sample(cx, cy);
      if (n > 3 || o.corners) for (let k = 0; k < Math.min(n, 4); k++) { const c = PI[s + k]; sample(cx + 0.7 * (CX[c] - cx), cy + 0.7 * (CY[c] - cy)); }
      if (seen) {
        const poly = []; for (let k = s; k < s + n; k++) poly.push([x2px(CX[PI[k]]), y2py(CY[PI[k]])]);
        res.push({ id: a, frac: seen / total, poly, center: [x2px(cx), y2py(cy)] });
      }
    }
    return { areas: res, total: nA, z: T.z };
  }

  function nearestArea(w, hint, snap) {
    const f = floorAt(w.x, w.y, typeof w.z === 'number' ? w.z : hint);
    if (f) return f.area;
    let best = -1, bd = snap;
    for (let a = 0; a < nA; a++) { const d = Math.hypot(ACX[a] - w.x, ACY[a] - w.y); if (d < bd) { bd = d; best = a; } }
    return best;
  }
  function path(a, b, o) {
    o = o || {};
    const A = wpt(a, o.hintA != null ? o.hintA : o.hint), B = wpt(b, o.hintB != null ? o.hintB : o.hint);
    if (!A || !B) return null;
    const sa = nearestArea(A, o.hint, 300), sb = nearestArea(B, o.hint, 300);
    if (sa < 0 || sb < 0) return null;
    // A* over the nav graph, edge cost = 3D distance between centroids (awpy PathWeight::Distance)
    const dist = new Float64Array(nA).fill(Infinity), prev = new Int32Array(nA).fill(-1), done = new Uint8Array(nA);
    const hp = [[0, sa]]; dist[sa] = 0;
    const hq = (x) => Math.hypot(ACX[x] - ACX[sb], ACY[x] - ACY[sb], ACZ[x] - ACZ[sb]);
    const push = (it) => { hp.push(it); let i = hp.length - 1; while (i > 0) { const pi = (i - 1) >> 1; if (hp[pi][0] <= hp[i][0]) break; [hp[pi], hp[i]] = [hp[i], hp[pi]]; i = pi; } };
    const pop = () => { const top = hp[0], last = hp.pop(); if (hp.length) { hp[0] = last; let i = 0; for (;;) { let l = 2 * i + 1, r = l + 1, m = i; if (l < hp.length && hp[l][0] < hp[m][0]) m = l; if (r < hp.length && hp[r][0] < hp[m][0]) m = r; if (m === i) break; [hp[m], hp[i]] = [hp[i], hp[m]]; i = m; } } return top; };
    hp.length = 0; push([hq(sa), sa]);
    while (hp.length) {
      const [, u] = pop();
      if (done[u]) continue; done[u] = 1;
      if (u === sb) break;
      for (let k = KS[u]; k < KS[u + 1]; k++) {
        const v = KI[k], nd = dist[u] + Math.hypot(ACX[u] - ACX[v], ACY[u] - ACY[v], ACZ[u] - ACZ[v]);
        if (nd < dist[v]) { dist[v] = nd; prev[v] = u; push([nd + hq(v), v]); }
      }
    }
    if (!done[sb]) return { found: false };
    const chain = []; for (let v = sb; v >= 0; v = prev[v]) chain.push(v); chain.reverse();
    // polyline: start, centroids of the areas in between, end
    const pts = [[A.x, A.y, A.z]];
    for (let i = 1; i < chain.length - 1; i++) pts.push([ACX[chain[i]], ACY[chain[i]], ACZ[chain[i]]]);
    pts.push([B.x, B.y, B.z]);
    const len3 = P => { let s = 0; for (let i = 1; i < P.length; i++) s += Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1], P[i][2] - P[i - 1][2]); return s; };
    const graphUnits = len3(pts);
    // smoothing: from each point jump to the farthest later point that can be walked in a straight line on the nav
    function walkable(P, Q) {
      const d = Math.hypot(Q[0] - P[0], Q[1] - P[1]), n = Math.ceil(d / 12);
      let z = P[2];
      for (let i = 1; i <= n; i++) {
        const t = i / n, x = P[0] + (Q[0] - P[0]) * t, y = P[1] + (Q[1] - P[1]) * t;
        const nz = floorNear(x, y, z, 22);
        if (nz === null) return false;
        z = nz;
      }
      return Math.abs(z - Q[2]) < 30;
    }
    const sm = [pts[0]];
    if (o.smooth !== false) {
      let i = 0;
      while (i < pts.length - 1) {
        let j = pts.length - 1;
        while (j > i + 1 && !walkable(pts[i], pts[j])) j--;
        sm.push(pts[j]); i = j;
      }
    } else for (let i = 1; i < pts.length; i++) sm.push(pts[i]);
    const units = len3(sm);
    let xy = 0; for (let i = 1; i < sm.length; i++) xy += Math.hypot(sm[i][0] - sm[i - 1][0], sm[i][1] - sm[i - 1][1]);
    return {
      found: true, points: sm.map(q => [x2px(q[0]), y2py(q[1]), q[2]]), units, unitsXY: xy, graphUnits,
      sec: { knife: units / SPEED.knife, rifle: units / SPEED.rifle }, speeds: SPEED, areas: chain.length,
    };
  }

  function zAt(px, py, hint) {
    const x = w2x(px), y = w2y(py);
    const f = floorAt(x, y, hint);
    if (!f) return null;
    const all = [];
    for (const a of areasAt(x, y)) { const z = zOn(a, x, y); all.push({ z, level: levelOf(z) }); }
    return { z: f.z, level: levelOf(f.z), area: f.area, snapped: f.snapped, floors: all };
  }

  function setMap(meta) { M = meta; }
  function stats() { return { tris: nT, nodes: nn, areas: nA, smokes: smokes.length }; }

  return {
    setMap, loadMesh, loadNav, zAt, visible, viewPolygon, seenFrom, smokeShape, addSmoke, removeSmoke, clearSmokes, path, stats,
    // for tests
    _occluded: occluded, _closest: closest,
  };
}

/* ---------- main thread ---------- */
(function (root) {
  'use strict';
  async function fetchBytes(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error('geo: ' + url + ' ' + r.status);
    const buf = await r.arrayBuffer();
    const u8 = new Uint8Array(buf);
    if (u8[0] === 0x1f && u8[1] === 0x8b) {                  // gzip (unless the server already decoded it)
      return await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    }
    return buf;
  }

  // The worker script: GeoCore, fetchBytes and a message loop. `load` fetches and decodes the two files inside the worker.
  function makeWorker() {
    const src = GeoCore.toString() + '\n' + fetchBytes.toString() + '\n' + [
      'const core = GeoCore();',
      'self.onmessage = async (e) => {',
      '  const { id, op, args } = e.data;',
      '  try {',
      '    let r;',
      '    if (op === "load") {',
      '      const t0 = performance.now();',
      '      const a = args[0];',
      '      const [mb, nb] = await Promise.all([fetchBytes(a.meshUrl), fetchBytes(a.navUrl)]);',
      '      const t1 = performance.now();',
      '      core.setMap(a.meta); core.loadNav(nb); const t2 = performance.now(); core.loadMesh(mb); const t3 = performance.now();',
      '      r = { fetchMs: t1 - t0, navMs: t2 - t1, meshBvhMs: t3 - t2, totalMs: t3 - t0, stats: core.stats() };',
      '    } else r = core[op].apply(null, args);',
      '    self.postMessage({ id, ok: true, r });',
      '  } catch (err) { self.postMessage({ id, ok: false, err: String(err && err.stack || err) }); }',
      '};',
    ].join('\n');
    return new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
  }

  class Geo {
    /**
     * @param {object} [o]  base: folder with the data files (default "geo/data/", relative to the page)
     */
    constructor(o) {
      o = o || {};
      this.base = new URL(o.base || 'geo/data/', root.location ? root.location.href : undefined).href;
      this.w = null; this.pending = new Map(); this.n = 1; this.meta = null; this.map = null;
      this.ready = Promise.reject(new Error('geo: call load(map) first')); this.ready.catch(() => {});
    }
    _call(op, args) {
      return new Promise((res, rej) => { const id = this.n++; this.pending.set(id, { res, rej }); this.w.postMessage({ id, op, args }); });
    }
    /**
     * Load a map. opts: { crop: [x0, y0] (the site's crop of the 1024 px radar; default: data/maps.json), scale, posX, posY }
     * Resolves (also as `ready`) with the timing of the load. Calling again with another map starts a fresh worker.
     */
    load(map, opts) {
      opts = opts || {};
      if (this.w) { this.w.terminate(); this.pending.forEach(p => p.rej(new Error('geo: reloaded'))); this.pending.clear(); }
      this.map = map;
      const w = this.w = makeWorker();
      w.onmessage = e => { const { id, ok, r, err } = e.data, p = this.pending.get(id); if (!p) return; this.pending.delete(id); ok ? p.res(r) : p.rej(new Error(err)); };
      w.onerror = e => { this.pending.forEach(p => p.rej(new Error('geo worker: ' + e.message))); this.pending.clear(); };
      const t0 = performance.now();
      this.ready = (async () => {
        const info = await (await fetch(this.base + 'maps.json')).json();
        const m = info[map]; if (!m) throw new Error('geo: no data for ' + map);
        const crop = opts.crop || m.crop;
        const meta = { posX: opts.posX != null ? opts.posX : m.posX, posY: opts.posY != null ? opts.posY : m.posY, scale: opts.scale != null ? opts.scale : m.scale, cx: crop[0], cy: crop[1], lowerBelowZ: m.lowerBelowZ != null ? m.lowerBelowZ : null };
        this.meta = meta;
        const r = await this._call('load', [{ meshUrl: this.base + m.mesh.file, navUrl: this.base + m.nav.file, meta }]);
        r.sinceCallMs = performance.now() - t0;
        return r;
      })();
      return this.ready;
    }
    // synchronous frame helpers (after `ready`)
    pxToWorld(px, py) { const m = this.meta; return [m.posX + (px + m.cx) * m.scale, m.posY - (py + m.cy) * m.scale]; }
    worldToPx(x, y) { const m = this.meta; return [(x - m.posX) / m.scale - m.cx, (m.posY - y) / m.scale - m.cy]; }
    /** floor under a pixel: { z, level: 'upper'|'lower'|null (Nuke), area, snapped, floors: [{z, level}] } or null. hint: floor z, 'upper' or 'lower'. */
    zAt(px, py, hint) { return this.ready.then(() => this._call('zAt', [px, py, hint])); }
    /** line of sight between two points at eye height (64; { crouch: true } = 46). opts: heightA, heightB, smokes (default true), hint. */
    visible(a, b, opts) { return this.ready.then(() => this._call('visible', [a, b, opts])); }
    /** { poly: [[px, py], ...], rays, z }: what a player at p facing angleDeg (board angle) sees within fovDeg and rangePx. opts: terrain (default true), smokes, crouch, stepDeg, refine, hint. */
    viewPolygon(p, angleDeg, fovDeg, rangePx, opts) { return this.ready.then(() => this._call('viewPolygon', [p, angleDeg, fovDeg, rangePx, opts])); }
    /** { areas: [{ id, frac, poly, center }], total }: nav areas from which a player standing there sees point p (target 46 above the floor). opts: rangePx, smokes, hint. */
    seenFrom(p, opts) { return this.ready.then(() => this._call('seenFrom', [p, opts])); }
    /** footprint of a smoke landing at p, not added: { poly, rings, center, radiusPx, cells }. */
    smokeShape(p, opts) { return this.ready.then(() => this._call('smokeShape', [p, opts])); }
    /** add a smoke as a blocker for visible / viewPolygon / seenFrom; returns the shape with an id. */
    addSmoke(p, opts) { return this.ready.then(() => this._call('addSmoke', [p, opts])); }
    removeSmoke(id) { return this.ready.then(() => this._call('removeSmoke', [id])); }
    clearSmokes() { return this.ready.then(() => this._call('clearSmokes', [])); }
    /** { found, points: [[px, py, z]], units, unitsXY, graphUnits, sec: { knife, rifle }, areas } */
    path(a, b, opts) { return this.ready.then(() => this._call('path', [a, b, opts])); }
    stats() { return this.ready.then(() => this._call('stats', [])); }
    terminate() { if (this.w) this.w.terminate(); this.w = null; }
  }
  Geo.SPEED = { knife: 250, rifle: 215 };
  root.Geo = Geo;
  if (typeof module !== 'undefined' && module.exports) module.exports = { Geo, GeoCore };
})(typeof self !== 'undefined' ? self : globalThis);

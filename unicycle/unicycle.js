/**
 * Unicycle robot viewer and control using MuJoCo WASM + Three.js.
 * Load unicycle.mjcf, run simulation, apply control.
 * Default: show the humanoid and enable LQR upright balance. When LQR is off, use the pad plus
 * W/S/↑/↓ for manual lean and wheel torque control. LQR uses mjd_transitionFD + DARE,
 * mj_differentiatePos for state error, and mj_jacBodyCom at linearization.
 */
import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/controls/OrbitControls.js";

// MJCF and meshes live next to this script under unicycle/assets/ (not repo-root assets/).
const ASSETS_URL = new URL("./assets/", import.meta.url).href;
const SIMPLIFIED_MJCF_URL = new URL("unicycle_simplified.xml", ASSETS_URL).href;
const FULL_MJCF_URL = new URL("unicycle_with_humanoid.xml", ASSETS_URL).href;
// Meshes are referenced only by the full humanoid MJCF; the simplified model is primitives only.
const MESH_FILES = ["seat.obj", "shaft.obj", "wheel_crank_mesh.obj"];

// Simplified-model body names whose rendered geoms should hide when the humanoid ragdoll is shown.
const HIDE_ON_RAGDOLL = new Set(["com", "seat", "wheel"]);
// Joints whose simplified qpos/qvel are pinned onto the visual humanoid each frame.
// After re-rooting both models at the wheel, these three joints exist with identical names,
// axes, and angle conventions in both models, so the pin is a direct qpos/qvel copy.
const PINNED_JOINTS = ["pelvis_y", "pelvis_x", "wheel"];
// Joints on the visual model that must stay at 0 so that the full humanoid's upper body
// (torso, waist_lower, pelvis) remains rigidly stacked above the pelvis hinge, exactly as
// in the simplified model — the simplified rider has no spine DOFs between the pelvis
// hinges and the wheel, so letting the abdomen triplet ragdoll would offset the visual
// upper body from the simplified one.
const ZERO_PINNED_JOINTS = ["abdomen_z", "abdomen_y", "abdomen_x"];

// Use mujoco.mjtGeom.*.value for geom type; do not hardcode (enum order: PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, ...).

// MuJoCo is Z-up; Three.js is Y-up. Map (mj_x, mj_y, mj_z) -> (mj_x, mj_z, -mj_y).
function getMujocoPos(data, bodyId, out) {
  const mx = data.xpos[bodyId * 3];
  const my = data.xpos[bodyId * 3 + 1];
  const mz = data.xpos[bodyId * 3 + 2];
  out.set(mx, mz, -my);
  return out;
}

// Quat: MuJoCo (w,x,y,z) -> Three.js (x,y,z,w) with Z-up to Y-up swizzle per zalo/mujoco_wasm.
function getMujocoQuat(data, bodyId, out) {
  const w = data.xquat[bodyId * 4];
  const x = data.xquat[bodyId * 4 + 1];
  const y = data.xquat[bodyId * 4 + 2];
  const z = data.xquat[bodyId * 4 + 3];
  out.set(-x, -z, y, -w);
  return out;
}

// Swizzle a mesh-local Z-up vertex/normal (mx,my,mz) into Three.js Y-up (mx, mz, -my).
// This matches the body/geom pose swizzle used in getMujocoPos/getMujocoQuat, and
// preserves winding order (proper rotation about +x).
function buildMeshGeometry(model, geomId) {
  const meshId = model.geom_dataid[geomId];
  if (meshId < 0) return new THREE.SphereGeometry(0.05, 8, 8);
  const vertadr = model.mesh_vertadr[meshId];
  const vertnum = model.mesh_vertnum[meshId];
  const faceadr = model.mesh_faceadr[meshId];
  const facenum = model.mesh_facenum[meshId];

  const positions = new Float32Array(vertnum * 3);
  for (let i = 0; i < vertnum; i++) {
    const src = (vertadr + i) * 3;
    const mx = model.mesh_vert[src];
    const my = model.mesh_vert[src + 1];
    const mz = model.mesh_vert[src + 2];
    positions[i * 3] = mx;
    positions[i * 3 + 1] = mz;
    positions[i * 3 + 2] = -my;
  }

  const indices = new Uint32Array(facenum * 3);
  for (let i = 0; i < facenum; i++) {
    indices[i * 3] = model.mesh_face[(faceadr + i) * 3];
    indices[i * 3 + 1] = model.mesh_face[(faceadr + i) * 3 + 1];
    indices[i * 3 + 2] = model.mesh_face[(faceadr + i) * 3 + 2];
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));
  if (model.mesh_normal && model.mesh_normal.length >= (vertadr + vertnum) * 3) {
    const normals = new Float32Array(vertnum * 3);
    for (let i = 0; i < vertnum; i++) {
      const src = (vertadr + i) * 3;
      const nx = model.mesh_normal[src];
      const ny = model.mesh_normal[src + 1];
      const nz = model.mesh_normal[src + 2];
      normals[i * 3] = nx;
      normals[i * 3 + 1] = nz;
      normals[i * 3 + 2] = -ny;
    }
    geom.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  } else {
    geom.computeVertexNormals();
  }
  return geom;
}

// Geometry creation aligned with zalo/mujoco_wasm: plane 100×100, cylinder Y-up (no extra rotation).
function createGeometry(mujoco, model, geomId) {
  const type = model.geom_type[geomId];
  const size = [
    model.geom_size[geomId * 3],
    model.geom_size[geomId * 3 + 1],
    model.geom_size[geomId * 3 + 2],
  ];
  const G = mujoco.mjtGeom || {};
  const plane = G.mjGEOM_PLANE?.value ?? 0;
  const sphere = G.mjGEOM_SPHERE?.value ?? 2;
  const capsule = G.mjGEOM_CAPSULE?.value ?? 3;
  const ellipsoid = G.mjGEOM_ELLIPSOID?.value ?? 4;
  const cylinder = G.mjGEOM_CYLINDER?.value ?? 5;
  const box = G.mjGEOM_BOX?.value ?? 6;
  const mesh = G.mjGEOM_MESH?.value ?? 7;
  let geometry;
  if (type === plane) {
    geometry = new THREE.PlaneGeometry(100, 100);
    geometry.rotateX(-Math.PI / 2);
  } else if (type === sphere) {
    geometry = new THREE.SphereGeometry(size[0], 24, 24);
  } else if (type === cylinder) {
    geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 24);
  } else if (type === capsule) {
    // MuJoCo capsule: size=[radius, half_length_of_cylindrical_section], axis = local +Z.
    // Three.js CapsuleGeometry(radius, length, capSegments, radialSegments) builds along +Y
    // and `length` is the length of the cylindrical middle section. After our Z-up→Y-up
    // quat swizzle (getMujocoQuat), a MuJoCo local +Z axis maps to Three's local +Y, so no
    // extra rotation is needed.
    const r = size[0];
    const len = Math.max(1e-6, size[1] * 2);
    geometry = THREE.CapsuleGeometry
      ? new THREE.CapsuleGeometry(r, len, 8, 24)
      : new THREE.CylinderGeometry(r, r, len, 24);
  } else if (type === box) {
    geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
  } else if (type === ellipsoid) {
    geometry = new THREE.SphereGeometry(1.0, 24, 24);
    geometry.scale(size[0], size[2], size[1]);
  } else if (type === mesh) {
    geometry = buildMeshGeometry(model, geomId);
  } else {
    geometry = new THREE.SphereGeometry(0.1, 8, 8);
  }
  return geometry;
}

// Geom pose: same position/quat swizzle as zalo/mujoco_wasm (getPosition / getQuaternion).
function createMeshForGeom(mujoco, model, geomId, material, isPlane) {
  const geometry = createGeometry(mujoco, model, geomId);
  const mesh = new THREE.Mesh(geometry, material);
  const gpos = model.geom_pos;
  if (gpos && geomId * 3 + 2 < gpos.length) {
    const mx = gpos[geomId * 3], my = gpos[geomId * 3 + 1], mz = gpos[geomId * 3 + 2];
    mesh.position.set(mx, mz, -my);
  }
  const gquat = model.geom_quat;
  if (gquat && geomId * 4 + 3 < gquat.length && !isPlane) {
    const w = gquat[geomId * 4], x = gquat[geomId * 4 + 1], y = gquat[geomId * 4 + 2], z = gquat[geomId * 4 + 3];
    mesh.quaternion.set(-x, -z, y, -w);
  }
  mesh.castShadow = !isPlane;
  mesh.receiveShadow = true;
  return mesh;
}

// ---------------- MuJoCo material/texture extraction -----------------------------------------
// Follows zalo/mujoco_wasm's mujocoUtils.js: per-geom appearance comes from
// `geom_matid` → `mat_{rgba,texid,texrepeat,specular,reflectance,shininess}`, and per-texture
// pixel data is read directly from `tex_data` (expanded to RGBA for Three.DataTexture).
//
// MuJoCo texture types (model.tex_type):   0 = 2D, 1 = cube, 2 = skybox.
// Texture roles (model.mat_texid is nmat × mjNTEXROLE): we use role 1 = mjTEXROLE_RGB.

const MJ_TEX_2D = 0;
const MJ_TEX_CUBE = 1;
const MJ_TEX_SKYBOX = 2;
const MJ_NTEXROLE = 10;
const MJ_TEXROLE_RGB = 1;

// Build a THREE.DataTexture for every texture in the model by slicing model.tex_data.
// For cube/skybox types the data is a vertical strip of 6 width×width faces stacked in MuJoCo's
// face order (right, left, up, down, front, back). We return the full strip as a single
// DataTexture (suitable for .map on primitives with standard UVs, matching zalo's approach);
// a separate helper splits the strip into a proper CubeTexture for the skybox.
function buildDataTextures(model) {
  const out = [];
  for (let t = 0; t < model.ntex; t++) {
    const width = model.tex_width[t];
    const height = model.tex_height[t];
    const channels = model.tex_nchannel[t];
    const adr = model.tex_adr[t];
    const rgba = new Uint8Array(width * height * 4);
    for (let p = 0; p < width * height; p++) {
      const src = adr + p * channels;
      rgba[p * 4] = model.tex_data[src];
      rgba[p * 4 + 1] = channels > 1 ? model.tex_data[src + 1] : rgba[p * 4];
      rgba[p * 4 + 2] = channels > 2 ? model.tex_data[src + 2] : rgba[p * 4];
      rgba[p * 4 + 3] = channels > 3 ? model.tex_data[src + 3] : 255;
    }
    const tex = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    out.push({ tex, width, height, channels, adr, type: model.tex_type[t] });
  }
  return out;
}

// Split a cube/skybox strip (width × 6*width) into a THREE.CubeTexture.
// MuJoCo face order in the strip: [+X, -X, +Z, -Z, +Y, -Y].
// Three.js CubeTexture order:     [+X, -X, +Y, -Y, +Z, -Z].
// We additionally remap for the MuJoCo-Z-up → Three-Y-up scene convention used everywhere else:
// Three  +Y  ← MuJoCo +Z (up)  : strip index 2
// Three  -Y  ← MuJoCo -Z (down): strip index 3
// Three  +Z  ← MuJoCo -Y (back in Three = -Y in MuJoCo, per (mx, mz, -my)): strip index 5
// Three  -Z  ← MuJoCo +Y       : strip index 4
function buildCubeFromStrip(model, texIndex) {
  const w = model.tex_width[texIndex];
  const h = model.tex_height[texIndex];
  if (h !== 6 * w) return null;
  const channels = model.tex_nchannel[texIndex];
  const adr = model.tex_adr[texIndex];
  const faces = [];
  for (let f = 0; f < 6; f++) {
    const cvs = document.createElement("canvas");
    cvs.width = w;
    cvs.height = w;
    const ctx = cvs.getContext("2d");
    const img = ctx.createImageData(w, w);
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        const src = adr + ((f * w + y) * w + x) * channels;
        const dst = (y * w + x) * 4;
        img.data[dst] = model.tex_data[src];
        img.data[dst + 1] = channels > 1 ? model.tex_data[src + 1] : img.data[dst];
        img.data[dst + 2] = channels > 2 ? model.tex_data[src + 2] : img.data[dst];
        img.data[dst + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    faces.push(cvs);
  }
  const order = [0, 1, 2, 3, 5, 4];
  const cube = new THREE.CubeTexture(order.map((i) => faces[i]));
  cube.colorSpace = THREE.SRGBColorSpace;
  cube.needsUpdate = true;
  return cube;
}

const MUJOCO_CDNS = [
  "https://unpkg.com/mujoco-js@0.0.7/dist/mujoco_wasm.js",
  "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js",
];

async function loadMujocoFromCDN() {
  let lastErr;
  for (const url of MUJOCO_CDNS) {
    try {
      const mod = await import(/* webpackIgnore: true */ url);
      const load = mod.default ?? mod;
      const m = typeof load === "function" ? await load() : load;
      if (m && m.FS && m.MjModel) return m;
      lastErr = new Error("mujoco-js did not expose FS/MjModel");
    } catch (e) {
      lastErr = e;
    }
  }
  const msg = lastErr?.message ?? String(lastErr);
  throw new Error(
    `Failed to load mujoco-js from CDN (tried ${MUJOCO_CDNS.length} URLs). ${msg}. ` +
      "If you see 'module.require', avoid esm.sh; use unpkg/jsdelivr directly."
  );
}

async function main() {
  // Avoid esm.sh: it injects Node polyfills (unenv) and mujoco-js hits module.require.
  const mujoco = await loadMujocoFromCDN();

  mujoco.FS.mkdir("/working");
  mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

  // Fetch both MJCFs + mesh OBJs in parallel and stage them in MEMFS.
  // MuJoCo resolves `<mesh file="foo.obj"/>` relative to the MJCF location when no meshdir is set.
  const [simpText, fullText, ...meshBuffers] = await Promise.all([
    fetch(SIMPLIFIED_MJCF_URL).then((r) => r.text()),
    fetch(FULL_MJCF_URL).then((r) => r.text()),
    ...MESH_FILES.map((name) =>
      fetch(new URL(name, ASSETS_URL)).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b))
    ),
  ]);
  mujoco.FS.writeFile("/working/unicycle.mjcf", simpText);
  mujoco.FS.writeFile("/working/humanoid.mjcf", fullText);
  MESH_FILES.forEach((name, i) => mujoco.FS.writeFile(`/working/${name}`, meshBuffers[i]));

  // Authoritative physics: simplified model.
  const model = mujoco.MjModel.loadFromXML("/working/unicycle.mjcf");
  const data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);

  // Visual-only: full humanoid. Disable all contacts so the ragdoll doesn't collide with anything.
  const model_v = mujoco.MjModel.loadFromXML("/working/humanoid.mjcf");
  for (let g = 0; g < model_v.ngeom; g++) {
    model_v.geom_contype[g] = 0;
    model_v.geom_conaffinity[g] = 0;
  }
  const data_v = new mujoco.MjData(model_v);
  mujoco.mj_forward(model_v, data_v);

  // Extract GPU textures directly from the MuJoCo model's procedurally-generated pixel data.
  // Both MJCFs declare the same asset set, so we only build them from the full model; either
  // source would produce identical data.
  const modelTextures = buildDataTextures(model_v);
  // Find the skybox texture (if any) for the scene background.
  let skyboxCube = null;
  for (let t = 0; t < modelTextures.length; t++) {
    if (modelTextures[t].type === MJ_TEX_SKYBOX) {
      skyboxCube = buildCubeFromStrip(model_v, t);
      break;
    }
  }

  // Resolve enum ids for mj_name2id; fall back to canonical constants if the wasm build
  // doesn't expose mjtObj. Canonical: mjOBJ_BODY=1, mjOBJ_JOINT=3, mjOBJ_ACTUATOR=19.
  const OBJ = mujoco.mjtObj || {};
  const OBJ_BODY = OBJ.mjOBJ_BODY?.value ?? 1;
  const OBJ_JOINT = OBJ.mjOBJ_JOINT?.value ?? 3;
  const OBJ_ACTUATOR = OBJ.mjOBJ_ACTUATOR?.value ?? 19;
  const jointAddr = (m, name) => {
    const id = mujoco.mj_name2id(m, OBJ_JOINT, name);
    if (id < 0) throw new Error(`joint not found: ${name}`);
    return { id, qposadr: m.jnt_qposadr[id], dofadr: m.jnt_dofadr[id] };
  };
  const actId = (m, name) => {
    const id = mujoco.mj_name2id(m, OBJ_ACTUATOR, name);
    if (id < 0) throw new Error(`actuator not found: ${name}`);
    return id;
  };
  const bodyId = (m, name) => {
    const id = mujoco.mj_name2id(m, OBJ_BODY, name);
    if (id < 0) throw new Error(`body not found: ${name}`);
    return id;
  };

  const ACT = {
    pelvis_y: actId(model, "pelvis_y"),
    pelvis_x: actId(model, "pelvis_x"),
    wheel: actId(model, "wheel"),
  };

  // qpos/qvel addresses for each pinned joint in both models.
  const pinMap = PINNED_JOINTS.map((name) => ({
    name,
    s: jointAddr(model, name),
    v: jointAddr(model_v, name),
  }));
  // Visual-model joints clamped to zero every frame (no counterpart in simplified).
  const zeroPinAddrs = ZERO_PINNED_JOINTS.map((name) => jointAddr(model_v, name));

  // Ctrlrange limits (radians) for the pelvis position servos.
  const PELVIS_Y_LO = model.actuator_ctrlrange[ACT.pelvis_y * 2];
  const PELVIS_Y_HI = model.actuator_ctrlrange[ACT.pelvis_y * 2 + 1];
  const PELVIS_X_LO = model.actuator_ctrlrange[ACT.pelvis_x * 2];
  const PELVIS_X_HI = model.actuator_ctrlrange[ACT.pelvis_x * 2 + 1];

  // ---------------- LQR (discrete-time, mjd_transitionFD + DARE) --------------------------------
  // State dimension: mjd_transitionFD uses x of size 2*nv+na; na=0 for this model.
  const NV = model.nv;
  const NQ = model.nq;
  const NU = model.nu;
  const NA = model.na || 0;
  const N_STATE = 2 * NV + NA;
  if (N_STATE !== 18) {
    console.warn(`LQR: expected n_state=18, got ${N_STATE} (nv=${NV}, na=${NA})`);
  }
  const WHEEL_TORQUE_MAX = 4;

  // Row-major: M[r*cols + c] = M_{r,c}. out = A (ar x ac) * B (ac x bc) -> (ar x bc)
  function matMul(A, ar, ac, B, br, bc, out) {
    for (let i = 0; i < ar; i++) {
      for (let j = 0; j < bc; j++) {
        let s = 0;
        for (let k = 0; k < ac; k++) s += A[i * ac + k] * B[k * bc + j];
        out[i * bc + j] = s;
      }
    }
  }
  /** Frobenius norm of n x m row-major. */
  function matFroNorm(M, n, m) {
    let s = 0;
    for (let i = 0, L = n * m; i < L; i++) s += M[i] * M[i];
    return Math.sqrt(s);
  }
  /**
   * Solve A X = B with A n x n, B n x p, all row-major, A destroyed (Gauss-Jordan with pivot).
   */
  function matSolve(A, b, n, p) {
    for (let col = 0; col < n; col++) {
      let pivot = col;
      let best = Math.abs(A[col * n + col]);
      for (let r = col + 1; r < n; r++) {
        const v = Math.abs(A[r * n + col]);
        if (v > best) {
          best = v;
          pivot = r;
        }
      }
      if (best < 1e-14) throw new Error("LQR: singular matrix in linear solve");
      if (pivot !== col) {
        for (let c = 0; c < n; c++) {
          const t = A[col * n + c];
          A[col * n + c] = A[pivot * n + c];
          A[pivot * n + c] = t;
        }
        for (let c = 0; c < p; c++) {
          const t = b[col * p + c];
          b[col * p + c] = b[pivot * p + c];
          b[pivot * p + c] = t;
        }
      }
      const inv = 1.0 / A[col * n + col];
      for (let c = 0; c < n; c++) A[col * n + c] *= inv;
      for (let c = 0; c < p; c++) b[col * p + c] *= inv;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = A[r * n + col];
        if (f === 0) continue;
        for (let c = 0; c < n; c++) A[r * n + c] -= f * A[col * n + c];
        for (let c2 = 0; c2 < p; c2++) b[r * p + c2] -= f * b[col * p + c2];
      }
    }
  }

  function makeDiag(Qdiag, n) {
    const P = new Float64Array(n * n);
    for (let i = 0; i < n; i++) P[i * n + i] = Qdiag[i];
    return P;
  }

  function matCopyToLen(dst, src, len) {
    for (let i = 0; i < len; i++) dst[i] = src[i];
  }

  function controllabilityRank(A, B, n, m) {
    const cols = n * m;
    const ctrb = new Float64Array(n * cols);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < m; c++) ctrb[r * cols + c] = B[r * m + c];
    }
    const block = new Float64Array(n * m);
    matCopyToLen(block, B, n * m);
    for (let p = 1; p < n; p++) {
      const next = new Float64Array(n * m);
      matMul(A, n, n, block, n, m, next);
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < m; c++) ctrb[r * cols + p * m + c] = next[r * m + c];
      }
      matCopyToLen(block, next, n * m);
    }
    let rank = 0;
    const eps = 1e-9;
    for (let col = 0; col < cols && rank < n; col++) {
      let pivot = rank;
      let best = Math.abs(ctrb[pivot * cols + col]);
      for (let r = rank + 1; r < n; r++) {
        const v = Math.abs(ctrb[r * cols + col]);
        if (v > best) {
          best = v;
          pivot = r;
        }
      }
      if (best < eps) continue;
      if (pivot !== rank) {
        for (let c = col; c < cols; c++) {
          const t = ctrb[rank * cols + c];
          ctrb[rank * cols + c] = ctrb[pivot * cols + c];
          ctrb[pivot * cols + c] = t;
        }
      }
      const inv = 1.0 / ctrb[rank * cols + col];
      for (let c = col; c < cols; c++) ctrb[rank * cols + c] *= inv;
      for (let r = 0; r < n; r++) {
        if (r === rank) continue;
        const f = ctrb[r * cols + col];
        if (Math.abs(f) < eps) continue;
        for (let c = col; c < cols; c++) ctrb[r * cols + c] -= f * ctrb[rank * cols + c];
      }
      rank++;
    }
    return rank;
  }

  /**
   * Riccati: P <- A'PA - A'PB (R + B'PB)^-1 B'PA + Q,  then K = (R + B'PB)^-1 B' P A
   * A, B, Qdiag, Rdiag, K as row-major. B is n x m.
   */
  function solveDARE(A, B, Qdiag, Rdiag, n, m) {
    const Atrans = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Atrans[i * n + j] = A[j * n + i];
    const P = makeDiag(Qdiag, n);
    const ApA = new Float64Array(n * n);
    const AtransP = new Float64Array(n * n);
    const ApB = new Float64Array(n * m);
    const PB = new Float64Array(n * m);
    const Smm = new Float64Array(m * m);
    const Swork = new Float64Array(m * m);
    const PA = new Float64Array(n * n);
    const TKn = new Float64Array(n * n);
    const Pprev = new Float64Array(n * n);
    const K = new Float64Array(m * n);
    for (let it = 0; it < 2000; it++) {
      for (let i = 0; i < n * n; i++) Pprev[i] = P[i];
      matMul(Atrans, n, n, P, n, n, AtransP);
      matMul(AtransP, n, n, A, n, n, ApA);
      matMul(AtransP, n, n, B, n, m, ApB);
      matMul(P, n, n, B, n, m, PB);
      for (let i = 0; i < m * m; i++) Smm[i] = 0;
      for (let r = 0; r < m; r++) for (let c = 0; c < m; c++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += B[k * m + r] * PB[k * m + c];
        Smm[r * m + c] = s;
      }
      for (let d = 0; d < m; d++) Smm[d * m + d] += Rdiag[d];
      matMul(P, n, n, A, n, n, PA);
      for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += B[k * m + r] * PA[k * n + c];
        K[r * n + c] = s;
      }
      matCopyToLen(Swork, Smm, m * m);
      matSolve(Swork, K, m, n);
      matMul(ApB, n, m, K, m, n, TKn);
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        const i = r * n + c;
        P[i] = ApA[i] - TKn[i] + (r === c ? Qdiag[r] : 0);
      }
      let diff = 0;
      for (let k = 0; k < n * n; k++) {
        const t = P[k] - Pprev[k];
        diff += t * t;
      }
      if (it > 30 && diff < 1e-20) break;
    }
    matMul(P, n, n, B, n, m, PB);
    for (let i = 0; i < m * m; i++) Smm[i] = 0;
    for (let r = 0; r < m; r++) for (let c = 0; c < m; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += B[k * m + r] * PB[k * m + c];
      Smm[r * m + c] = s;
    }
    for (let d = 0; d < m; d++) Smm[d * m + d] += Rdiag[d];
    matMul(P, n, n, A, n, n, PA);
    for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += B[k * m + r] * PA[k * n + c];
      K[r * n + c] = s;
    }
    matCopyToLen(Swork, Smm, m * m);
    matSolve(Swork, K, m, n);
    return { P, K };
  }

  function vecNormSq(v, len = v.length) {
    let s = 0;
    for (let i = 0; i < len; i++) s += v[i] * v[i];
    return s;
  }

  function pitchQuat(theta, out, offset = 0) {
    const half = 0.5 * theta;
    out[offset + 0] = Math.cos(half);
    out[offset + 1] = 0;
    out[offset + 2] = Math.sin(half);
    out[offset + 3] = 0;
  }

  function cleanDriftColumns(A, n, driftIndices) {
    const out = new Float64Array(A.length);
    matCopyToLen(out, A, A.length);
    for (let i = 0; i < driftIndices.length; i++) {
      const di = driftIndices[i];
      for (let r = 0; r < n; r++) out[r * n + di] = 0;
      out[di * n + di] = 1;
    }
    return out;
  }

  function solveDiscountedDARE(A, B, Qdiag, Rdiag, n, m, gamma, driftIndices) {
    const scale = Math.sqrt(gamma);
    const Aclean = cleanDriftColumns(A, n, driftIndices);
    const Ad = new Float64Array(Aclean.length);
    const Bd = new Float64Array(B.length);
    for (let i = 0; i < Aclean.length; i++) Ad[i] = scale * Aclean[i];
    for (let i = 0; i < B.length; i++) Bd[i] = scale * B[i];
    return solveDARE(Ad, Bd, Qdiag, Rdiag, n, m);
  }

  // State vector convention (from mjd_transitionFD): x = [dq (nv tangent), v (nv)].
  // Indices (nv = 9):
  //   0: world x (drift), 1: world y (drift), 2: z height,
  //   3: wx / roll, 4: wy / pitch, 5: wz / yaw (drift),
  //   6: wheel joint angle (pitch-like DOF between rider and wheel; NOT a symmetry),
  //   7: pelvis_y, 8: pelvis_x,
  //   9..17: velocities in the same order.
  // LQR weights heavy on pitch/roll + wheel-joint angle, moderate on wheel rate (to suppress
  // gyroscopic runaway), light on pelvis position servos. Drifts (x, y, yaw) get 1e-6
  // regularization so the discounted DARE is well-conditioned.
  const LQR_Q_ARR = [
    1e-6, 1e-6, 10.0, 900.0, 500.0, 1e-6, 220.0, 24.0, 90.0,
    1.0, 1.0, 5.0, 60.0, 30.0, 0.5, 34.0, 6.0, 18.0,
  ];
  const LQR_Q = new Float64Array(N_STATE);
  for (let i = 0; i < N_STATE; i++) LQR_Q[i] = i < LQR_Q_ARR.length ? LQR_Q_ARR[i] : 0;
  const LQR_R = new Float64Array(3);
  LQR_R[0] = 0.5; LQR_R[1] = 0.25; LQR_R[2] = 0.1;
  const LQR_GAMMA = 0.995;
  const LQR_DRIFT_INDICES = [0, 1, 5];
  const TRIM_DIM = 6;
  const TRIM_RES_DIM = 7;
  const LQR_FD_EPS = 1e-5;
  const LQR_LAT_ROLL_K = 0.0;
  const LQR_LAT_ROLLRATE_K = 0.0;
  const LQR_LAT_PELVISX_K = 0.0;
  const LQR_LAT_PELVISX_RATE_K = 0.0;
  // Similar to the numerical setup used in MuJoCo system identification workflows: solve a
  // small finite-difference least-squares problem in situ, then linearize at that operating
  // point. The current exported design remains as a safety fallback if the browser-side solve
  // fails on a particular machine.
  const TRIM_INIT_ARR = [0.26678, 0.01142, 0.00969, -0.02111, -0.00298, 0.0];
  const TRIM_LO_ARR = [0.24, -0.4, -0.8, -1.2, -1.309, -10.0];
  const TRIM_HI_ARR = [0.30, 0.4, 0.8, 0.5, 0.523599, 10.0];
  const FALLBACK_EQ_QPOS_ARR = [
    -0.0131,
    0.0,
    0.2667827733476135,
    0.9999836991915706,
    0.0,
    0.005709759289365526,
    0.0,
    0.009688113751446575,
    -0.021107694379710443,
    0.0,
  ];
  const FALLBACK_EQ_CTRL_ARR = [-0.0029829678750017824, 0.0, 0.0];
  const FALLBACK_LQR_K_ARR = [
    [
      6.13542781064893e-10, -1.5698676034645377e-19, 7.430468793395355, -4.886489635827333e-09,
      2.881043893298483, -5.978483696927459e-11, -2.3066854057097426, -0.4829799813314789,
      9.878798347466043e-10, 1.3062980902551564, 1.4330150508211037e-09, 0.1684701752038428,
      -1.2797739906621639e-09, 0.8348769390409227, -4.448754822095101e-12, -0.8497668579680243,
      0.1674877879508009, 3.7767551559723253e-10,
    ],
    [
      2.833506706141962e-10, 2.661164023988832e-09, 8.548430037241467e-10, 130.2434663085651,
      3.868803680961963e-09, 1.4873904196711452, -3.769314638648712e-09, -1.0577579790712226e-09,
      -26.54910251055881, 1.3668732639122796e-09, -38.801541126802285, -6.198385793722865e-12,
      33.948837878825145, 1.238566888723436e-09, 0.38769834509940515, -1.2106689473300328e-09,
      -3.720029137346446e-10, -9.638362148817997,
    ],
    [
      -4.2138923750862186e-08, -2.4665215213956814e-18, -0.7162437100377427, 1.106662917472118e-07,
      -601.2744147048007, 2.085634209521265e-09, 561.8268952070791, 166.39585394898873,
      -2.105072556398877e-08, -184.85160699414482, -3.233216465563306e-08, -0.012435455368709216,
      2.88967029093319e-08, -195.0144982532972, -1.998608282103436e-09, 198.87339167220892,
      62.90603597884151, -8.569243824786959e-09,
    ],
  ];

  const EQ_QPOS = new Float64Array(NQ);
  const EQ_CTRL = new Float64Array(NU);
  const lqrK = new Float64Array(NU * N_STATE);
  let lqrCtrbRank = 0;
  let lqrTrimCost = 0;
  let lqrSource = "dynamic";

  const trimVars = new Float64Array(TRIM_DIM);
  const trimLo = new Float64Array(TRIM_LO_ARR);
  const trimHi = new Float64Array(TRIM_HI_ARR);
  for (let i = 0; i < TRIM_DIM; i++) trimVars[i] = TRIM_INIT_ARR[i];

  const trimData = new mujoco.MjData(model);
  const trimResidual = new Float64Array(TRIM_RES_DIM);
  const trimCandidate = new Float64Array(TRIM_DIM);
  const trimResidualCand = new Float64Array(TRIM_RES_DIM);
  const trimResidualPlus = new Float64Array(TRIM_RES_DIM);
  const trimResidualMinus = new Float64Array(TRIM_RES_DIM);
  const trimJ = new Float64Array(TRIM_RES_DIM * TRIM_DIM);
  const trimH = new Float64Array(TRIM_DIM * TRIM_DIM);
  const trimG = new Float64Array(TRIM_DIM);
  const trimStep = new Float64Array(TRIM_DIM);
  function applyTrimVarsToData(dst, vars) {
    for (let i = 0; i < NQ; i++) dst.qpos[i] = 0;
    for (let i = 0; i < NV; i++) dst.qvel[i] = 0;
    for (let i = 0; i < NU; i++) dst.ctrl[i] = 0;
    dst.qpos[0] = -0.0131;
    dst.qpos[2] = vars[0];
    pitchQuat(vars[1], dst.qpos, 3);
    dst.qpos[7] = vars[2];
    dst.qpos[8] = vars[3];
    dst.qpos[9] = 0;
    dst.ctrl[0] = vars[4];
    dst.ctrl[1] = 0;
    dst.ctrl[2] = vars[5];
    mujoco.mj_forward(model, dst);
  }

  function setStateCtrl(dst, qposSrc, qvelSrc, ctrlSrc) {
    for (let i = 0; i < NQ; i++) dst.qpos[i] = qposSrc[i];
    for (let i = 0; i < NV; i++) dst.qvel[i] = qvelSrc[i];
    for (let i = 0; i < NU; i++) dst.ctrl[i] = ctrlSrc[i];
    mujoco.mj_forward(model, dst);
  }

  const lqrTmpDq = new Float64Array(NV);
  function stateErrorFromTrim(dst, qposRef, qvelRef, out) {
    mujoco.mj_differentiatePos(model, lqrTmpDq, 1.0, qposRef, dst.qpos);
    for (let i = 0; i < NV; i++) out[i] = lqrTmpDq[i];
    for (let i = 0; i < NV; i++) out[NV + i] = dst.qvel[i] - qvelRef[i];
  }

  function stateFromTangentPerturbation(qposRef, qvelRef, dx, qposOut, qvelOut) {
    for (let i = 0; i < NQ; i++) qposOut[i] = qposRef[i];
    for (let i = 0; i < NV; i++) {
      lqrTmpDq[i] = dx[i];
      qvelOut[i] = qvelRef[i] + dx[NV + i];
    }
    mujoco.mj_integratePos(model, qposOut, lqrTmpDq, 1.0);
  }

  function computeTrimResidual(vars, out) {
    applyTrimVarsToData(trimData, vars);
    out[0] = trimData.qacc[0];
    out[1] = trimData.qacc[2];
    out[2] = trimData.qacc[4];
    out[3] = trimData.qacc[6];
    out[4] = trimData.qacc[7];
    out[5] = vars[1] + vars[2] + vars[3];
    out[6] = trimData.ncon > 0 ? 0 : 10;
    return vecNormSq(out, TRIM_RES_DIM);
  }

  function buildTrimJacobian(vars, baseResidual) {
    for (let i = 0; i < trimJ.length; i++) trimJ[i] = 0;
    for (let j = 0; j < TRIM_DIM; j++) {
      const xj = vars[j];
      const step = Math.max(1e-6, 1e-4 * Math.max(1, Math.abs(xj)));
      const plus = Math.min(trimHi[j], xj + step);
      const minus = Math.max(trimLo[j], xj - step);
      if (plus === xj && minus === xj) continue;
      for (let k = 0; k < TRIM_DIM; k++) trimCandidate[k] = vars[k];
      let denom;
      if (minus !== xj) {
        trimCandidate[j] = minus;
        computeTrimResidual(trimCandidate, trimResidualMinus);
        trimCandidate[j] = plus;
        computeTrimResidual(trimCandidate, trimResidualPlus);
        denom = plus - minus;
        for (let r = 0; r < TRIM_RES_DIM; r++) {
          trimJ[r * TRIM_DIM + j] = (trimResidualPlus[r] - trimResidualMinus[r]) / denom;
        }
      } else {
        trimCandidate[j] = plus;
        computeTrimResidual(trimCandidate, trimResidualPlus);
        denom = plus - xj;
        for (let r = 0; r < TRIM_RES_DIM; r++) {
          trimJ[r * TRIM_DIM + j] = (trimResidualPlus[r] - baseResidual[r]) / denom;
        }
      }
    }
  }

  function solveTrimInJs(vars) {
    let lambda = 1e-3;
    let cost = computeTrimResidual(vars, trimResidual);
    for (let it = 0; it < 40; it++) {
      buildTrimJacobian(vars, trimResidual);
      for (let i = 0; i < trimH.length; i++) trimH[i] = 0;
      for (let i = 0; i < TRIM_DIM; i++) trimG[i] = 0;
      for (let r = 0; r < TRIM_RES_DIM; r++) {
        for (let i = 0; i < TRIM_DIM; i++) {
          const ji = trimJ[r * TRIM_DIM + i];
          trimG[i] += ji * trimResidual[r];
          for (let j = 0; j < TRIM_DIM; j++) trimH[i * TRIM_DIM + j] += ji * trimJ[r * TRIM_DIM + j];
        }
      }
      for (let i = 0; i < TRIM_DIM; i++) {
        trimH[i * TRIM_DIM + i] += lambda;
        trimStep[i] = -trimG[i];
      }
      const Hwork = new Float64Array(trimH);
      const stepWork = new Float64Array(trimStep);
      try {
        matSolve(Hwork, stepWork, TRIM_DIM, 1);
      } catch (e) {
        lambda *= 10;
        continue;
      }
      let stepNorm = 0;
      for (let i = 0; i < TRIM_DIM; i++) {
        const v = Math.max(trimLo[i], Math.min(trimHi[i], vars[i] + stepWork[i]));
        trimCandidate[i] = v;
        const d = v - vars[i];
        stepNorm += d * d;
      }
      const candCost = computeTrimResidual(trimCandidate, trimResidualCand);
      if (candCost < cost) {
        for (let i = 0; i < TRIM_DIM; i++) vars[i] = trimCandidate[i];
        matCopyToLen(trimResidual, trimResidualCand, TRIM_RES_DIM);
        cost = candCost;
        lambda = Math.max(1e-9, lambda * 0.3);
        if (stepNorm < 1e-16 || cost < 1e-18) break;
      } else {
        lambda = Math.min(1e6, lambda * 10);
      }
    }
    return cost;
  }

  function linearizeManualOneStep(qposRef, ctrlRef, Aout, Bout, eps = LQR_FD_EPS) {
    const plusData = new mujoco.MjData(model);
    const minusData = new mujoco.MjData(model);
    const qvelRef = new Float64Array(NV);
    const xPlus = new Float64Array(N_STATE);
    const xMinus = new Float64Array(N_STATE);
    const qposPlus = new Float64Array(NQ);
    const qvelPlus = new Float64Array(NV);
    const qposMinus = new Float64Array(NQ);
    const qvelMinus = new Float64Array(NV);
    const dx = new Float64Array(N_STATE);
    const ctrlPlus = new Float64Array(NU);
    const ctrlMinus = new Float64Array(NU);

    for (let i = 0; i < Aout.length; i++) Aout[i] = 0;
    for (let i = 0; i < Bout.length; i++) Bout[i] = 0;
    for (let col = 0; col < N_STATE; col++) {
      for (let i = 0; i < N_STATE; i++) dx[i] = 0;
      dx[col] = eps;
      stateFromTangentPerturbation(qposRef, qvelRef, dx, qposPlus, qvelPlus);
      setStateCtrl(plusData, qposPlus, qvelPlus, ctrlRef);
      mujoco.mj_step(model, plusData);
      stateErrorFromTrim(plusData, qposRef, qvelRef, xPlus);
      dx[col] = -eps;
      stateFromTangentPerturbation(qposRef, qvelRef, dx, qposMinus, qvelMinus);
      setStateCtrl(minusData, qposMinus, qvelMinus, ctrlRef);
      mujoco.mj_step(model, minusData);
      stateErrorFromTrim(minusData, qposRef, qvelRef, xMinus);
      for (let row = 0; row < N_STATE; row++) {
        Aout[row * N_STATE + col] = (xPlus[row] - xMinus[row]) / (2 * eps);
      }
    }

    for (let col = 0; col < NU; col++) {
      stateFromTangentPerturbation(qposRef, qvelRef, dx.fill(0), qposPlus, qvelPlus);
      for (let i = 0; i < NU; i++) {
        ctrlPlus[i] = ctrlRef[i];
        ctrlMinus[i] = ctrlRef[i];
      }
      ctrlPlus[col] += eps;
      ctrlMinus[col] -= eps;
      setStateCtrl(plusData, qposPlus, qvelPlus, ctrlPlus);
      mujoco.mj_step(model, plusData);
      stateErrorFromTrim(plusData, qposRef, qvelRef, xPlus);
      setStateCtrl(minusData, qposPlus, qvelPlus, ctrlMinus);
      mujoco.mj_step(model, minusData);
      stateErrorFromTrim(minusData, qposRef, qvelRef, xMinus);
      for (let row = 0; row < N_STATE; row++) {
        Bout[row * NU + col] = (xPlus[row] - xMinus[row]) / (2 * eps);
      }
    }
  }

  function loadFallbackLqrDesign(reason) {
    console.warn("LQR: dynamic solve failed, using exported fallback.", reason);
    for (let i = 0; i < NQ; i++) EQ_QPOS[i] = i < FALLBACK_EQ_QPOS_ARR.length ? FALLBACK_EQ_QPOS_ARR[i] : 0;
    for (let i = 0; i < NU; i++) EQ_CTRL[i] = i < FALLBACK_EQ_CTRL_ARR.length ? FALLBACK_EQ_CTRL_ARR[i] : 0;
    for (let r = 0; r < NU; r++) {
      for (let c = 0; c < N_STATE; c++) lqrK[r * N_STATE + c] = FALLBACK_LQR_K_ARR[r][c];
    }
    lqrCtrbRank = 14;
    lqrTrimCost = 0;
    lqrSource = "fallback";
  }

  try {
    lqrTrimCost = solveTrimInJs(trimVars);
    applyTrimVarsToData(trimData, trimVars);
    for (let i = 0; i < NQ; i++) EQ_QPOS[i] = trimData.qpos[i];
    for (let i = 0; i < NU; i++) EQ_CTRL[i] = trimData.ctrl[i];

    const lqrA = new Float64Array(N_STATE * N_STATE);
    const lqrB = new Float64Array(N_STATE * NU);
    let linearizationSource = "native";
    try {
      const lqrCdummy = new Float64Array(0);
      const lqrDdummy = new Float64Array(0);
      mujoco.mjd_transitionFD(model, trimData, 1e-6, 1, lqrA, lqrB, lqrCdummy, lqrDdummy);
      const nativeRank = controllabilityRank(lqrA, lqrB, N_STATE, NU);
      if (nativeRank < 10) throw new Error(`native rank ${nativeRank}/${N_STATE}`);
      lqrCtrbRank = nativeRank;
    } catch (nativeErr) {
      linearizeManualOneStep(EQ_QPOS, EQ_CTRL, lqrA, lqrB);
      lqrCtrbRank = controllabilityRank(lqrA, lqrB, N_STATE, NU);
      linearizationSource = "manual";
      if (lqrCtrbRank < 10) {
        throw new Error(`manual rank ${lqrCtrbRank}/${N_STATE} after native failure: ${nativeErr}`);
      }
    }
    const sol = solveDiscountedDARE(lqrA, lqrB, LQR_Q, LQR_R, N_STATE, NU, LQR_GAMMA, LQR_DRIFT_INDICES);
    matCopyToLen(lqrK, sol.K, sol.K.length);
    const kNorm = matFroNorm(lqrK, NU, N_STATE);
    if (!Number.isFinite(kNorm) || lqrCtrbRank < 10 || kNorm < 1e-6) {
      throw new Error(`degenerate dynamic LQR design (rank=${lqrCtrbRank}/${N_STATE}, |K|=${kNorm})`);
    }
    lqrSource = `dynamic-${linearizationSource}`;
  } catch (e) {
    loadFallbackLqrDesign(e);
  }

  // These arrays are now computed dynamically at startup in JS; they remain allocated so the
  // runtime controller and reset path can reuse the solved operating point and feedback gain.
  const lqrJacP = new Float64Array(3 * NV);
  const lqrJacR = new Float64Array(3 * NV);
  for (let i = 0; i < NQ; i++) trimData.qpos[i] = EQ_QPOS[i];
  for (let i = 0; i < NV; i++) trimData.qvel[i] = 0;
  for (let i = 0; i < NU; i++) trimData.ctrl[i] = EQ_CTRL[i];
  mujoco.mj_forward(model, trimData);
  for (let i = 0; i < NQ; i++) data.qpos[i] = EQ_QPOS[i];
  for (let i = 0; i < NV; i++) data.qvel[i] = 0;
  for (let i = 0; i < NU; i++) data.ctrl[i] = EQ_CTRL[i];
  mujoco.mj_forward(model, data);
  mujoco.mj_jacBodyCom(model, trimData, lqrJacP, lqrJacR, bodyId(model, "com"));
  const fj = matFroNorm(lqrJacP, 3, NV) + matFroNorm(lqrJacR, 3, NV);
  console.log(
    `LQR: ${lqrSource} rank = ${lqrCtrbRank}/${N_STATE}`,
    "|K|_F =", matFroNorm(lqrK, NU, N_STATE).toFixed(4),
    "mj_jacBodyCom|_F =", fj.toFixed(4),
    "trim cost =", lqrTrimCost.toExponential(2),
    "trim ctrl =", Array.from(EQ_CTRL).map((v) => v.toFixed(4)).join(", "),
  );
  mujoco.mj_resetData(model, data);
  mujoco.mj_forward(model, data);

  // Both the simplified and the full humanoid MJCFs are rooted at the wheel (free joint),
  // so the first 7 qpos values and the first 6 qvel values describe the wheel's world
  // pose/twist with identical semantics in both models. No offset computation is needed.

  const container = document.getElementById("unicycle-container") || document.body;
  const width = container.clientWidth || window.innerWidth;
  const height = container.clientHeight || window.innerHeight;

  const scene = new THREE.Scene();
  // Background: MuJoCo-declared skybox if present, otherwise a soft sky gradient color.
  const fallbackBg = new THREE.Color(0.55, 0.70, 0.85);
  if (skyboxCube) {
    scene.background = skyboxCube;
    // Use the skybox as an environment map too — cheap IBL for the physical materials,
    // which makes the rubbery/plastic parts of the unicycle pick up sky tones instead of
    // looking flat. Mirrors what zalo/mujoco_wasm does when a skybox is present.
    scene.environment = skyboxCube;
  } else {
    scene.background = fallbackBg;
  }
  // Distance fog blends props with the sky in the far field, matching the zalo demo.
  scene.fog = new THREE.Fog(skyboxCube ? new THREE.Color(0.55, 0.70, 0.85) : fallbackBg, 12, 40);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 100);
  camera.position.set(1.8, 1.3, 1.8);
  scene.add(camera);

  // Three-light rig (hemisphere bounce + key directional sun + filler spotlight).
  // Intensities are tuned for modern three.js (post-r165, no useLegacyLights) where
  // light units are physically scaled; MuJoCo-demo-style visuals need roughly π× the
  // pre-r155 values we had before.
  const hemi = new THREE.HemisphereLight(0xbcd7ff, 0x3a2e1d, 0.85);
  scene.add(hemi);
  const ambient = new THREE.AmbientLight(0xffffff, 0.25);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xfff2d8, 2.8);
  dirLight.position.set(4, 6, 4);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  dirLight.shadow.camera.near = 0.1;
  dirLight.shadow.camera.far = 25;
  dirLight.shadow.camera.left = -8;
  dirLight.shadow.camera.right = 8;
  dirLight.shadow.camera.top = 8;
  dirLight.shadow.camera.bottom = -8;
  dirLight.shadow.bias = -0.0004;
  dirLight.shadow.normalBias = 0.02;
  dirLight.shadow.radius = 3;
  scene.add(dirLight);
  const spot = new THREE.SpotLight(0xffffff, 7.0, 14, 1.1, 0.55, 1.0);
  spot.position.set(0, 3.5, 3);
  spot.target.position.set(0, 0.6, 0);
  spot.castShadow = true;
  spot.shadow.mapSize.width = 1024;
  spot.shadow.mapSize.height = 1024;
  spot.shadow.camera.near = 0.2;
  spot.shadow.camera.far = 14;
  spot.shadow.bias = -0.0004;
  spot.shadow.radius = 3;
  scene.add(spot);
  scene.add(spot.target);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  // ACES filmic gives the scene a punchier, more cinematic roll-off without crushing
  // highlights; exposure ~1.1 compensates for the filmic dip on neutral mid-tones.
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  container.appendChild(renderer.domElement);

  // ---------------- Target beacon (green disk + vertical beam + point light) -----------------
  // Goal: give the rider a visible nearby waypoint. The ground disk + additive ring + halo
  // reproduce the classic "landing pad" look; the vertical beam uses a custom shader that
  // fades alpha quadratically with height and pulses, so it reads as a sky-shot volumetric
  // column under additive blending without needing post-processing bloom.
  const TARGET_COLOR = new THREE.Color(0x3dff8c);
  const targetGroup = new THREE.Group();
  targetGroup.name = "unicycle-target";
  scene.add(targetGroup);

  const groundDisk = new THREE.Mesh(
    new THREE.CircleGeometry(0.32, 64),
    new THREE.MeshStandardMaterial({
      color: 0x052915,
      emissive: TARGET_COLOR,
      emissiveIntensity: 1.8,
      roughness: 0.6,
      metalness: 0.0,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
    }),
  );
  groundDisk.rotation.x = -Math.PI / 2;
  groundDisk.position.y = 0.005;
  targetGroup.add(groundDisk);

  const glowRing = new THREE.Mesh(
    new THREE.RingGeometry(0.33, 0.48, 64),
    new THREE.MeshBasicMaterial({
      color: TARGET_COLOR,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      toneMapped: false,
    }),
  );
  glowRing.rotation.x = -Math.PI / 2;
  glowRing.position.y = 0.007;
  targetGroup.add(glowRing);

  const halo = new THREE.Mesh(
    new THREE.CircleGeometry(0.95, 64),
    new THREE.MeshBasicMaterial({
      color: TARGET_COLOR,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    }),
  );
  halo.rotation.x = -Math.PI / 2;
  halo.position.y = 0.004;
  targetGroup.add(halo);

  // Custom shader for the beam: height-fading alpha (uv.y goes 0 at the base → 1 at the top
  // on a standard THREE.CylinderGeometry side), plus a slow pulse. toneMapped is disabled so
  // the additive contribution survives the ACES roll-off and stays luminous.
  const beamUniforms = { uTime: { value: 0 }, uColor: { value: TARGET_COLOR } };
  const beamMaterial = new THREE.ShaderMaterial({
    uniforms: beamUniforms,
    vertexShader: [
      "varying float vH;",
      "void main() {",
      "  vH = uv.y;",
      "  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);",
      "}",
    ].join("\n"),
    fragmentShader: [
      "uniform float uTime;",
      "uniform vec3 uColor;",
      "varying float vH;",
      "void main() {",
      "  float alpha = pow(1.0 - vH, 2.2);",
      "  float pulse = 0.80 + 0.20 * sin(uTime * 3.5);",
      "  vec3 col = uColor * (1.6 + pulse * 0.5);",
      "  gl_FragColor = vec4(col, alpha * pulse * 0.85);",
      "}",
    ].join("\n"),
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
  const BEAM_H = 2.6;
  const beamCore = new THREE.Mesh(
    new THREE.CylinderGeometry(0.07, 0.07, BEAM_H, 24, 1, true),
    beamMaterial,
  );
  beamCore.position.y = BEAM_H / 2;
  targetGroup.add(beamCore);
  const beamOuter = new THREE.Mesh(
    new THREE.CylinderGeometry(0.24, 0.30, BEAM_H, 24, 1, true),
    beamMaterial,
  );
  beamOuter.position.y = BEAM_H / 2;
  targetGroup.add(beamOuter);

  // Local point light at the disk: softly lights nearby props (floor, unicycle wheel)
  // with the target color so the beacon visibly bleeds onto the scene.
  const beaconLight = new THREE.PointLight(TARGET_COLOR, 2.4, 3.5, 2.0);
  beaconLight.position.set(0, 0.25, 0);
  targetGroup.add(beaconLight);

  // Spawn the target in a ring around the origin. 1.5 m keeps it clear of the starting
  // pose of the unicycle, 3.5 m keeps it on-camera without zooming out.
  const TARGET_MIN_R = 1.5;
  const TARGET_MAX_R = 3.5;
  const TARGET_REACH_R = 0.45;
  function respawnTarget() {
    const theta = Math.random() * Math.PI * 2;
    const r = TARGET_MIN_R + Math.random() * (TARGET_MAX_R - TARGET_MIN_R);
    // Three.js coords: ground plane is XZ, Y is up.
    targetGroup.position.set(Math.cos(theta) * r, 0, Math.sin(theta) * r);
  }
  respawnTarget();

  // Wheel body id (simplified model) for the reached-target check each frame.
  const S_WHEEL_BID = bodyId(model, "wheel");

  const keys = { wheelFwd: false, wheelBwd: false };
  const WHEEL_TORQUE = 4;
  let lqrEnabled = false;

  // LQR buffers: minimal-coordinate position difference (nv) + velocity error = state (2*nv).
  const lqrDq = new Float64Array(NV);
  const lqrQvelRef = new Float64Array(NV);
  const lqrQposRef = new Float64Array(NQ);
  const lqrDx = new Float64Array(N_STATE);
  // Normalized pad offset in [-1, 1]; 0 when the pad is idle.
  let padNormX = 0;
  let padNormY = 0;
  let padActive = false;

  function yawFromQpos4(w, x, y, z) {
    return Math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  }

  function applyLqr() {
    const q = data.qpos;
    const v = data.qvel;
    const psi = yawFromQpos4(q[3], q[4], q[5], q[6]);
    // Reference pose: equilibrium z/pitch/pelvis/wheel-joint, x/y/yaw copied from current state
    // so we do not try to control the pure symmetry drifts.
    lqrQposRef.set(EQ_QPOS);
    lqrQposRef[0] = q[0];
    lqrQposRef[1] = q[1];
    // Compose yaw(psi) * pitch(pitch_eq) as a Hamilton quaternion; this keeps the roll/pitch
    // reference aligned with the linearization trim and lets yaw slide freely.
    const ch = Math.cos(psi * 0.5);
    const sh = Math.sin(psi * 0.5);
    const pw = EQ_QPOS[3];
    const py = EQ_QPOS[5];
    lqrQposRef[3] = ch * pw;
    lqrQposRef[4] = -sh * py;
    lqrQposRef[5] = ch * py;
    lqrQposRef[6] = sh * pw;
    lqrQvelRef.fill(0);
    // mj_differentiatePos(m, out, dt, qpos1, qpos2) gives out = (qpos2 - qpos1) / dt in the
    // tangent space. mjd_transitionFD's state uses dq = (q - q_ref) in tangent, so pass
    // (qpos1 = qposRef, qpos2 = q) to get the correctly-signed linearization coordinate.
    mujoco.mj_differentiatePos(model, lqrDq, 1.0, lqrQposRef, q);
    // Zero out pure symmetry drifts (world x, y, yaw) — these are uncontrollable and their
    // columns in A were zeroed during DARE, so K has no meaningful gain on them either.
    lqrDq[0] = 0; lqrDq[1] = 0; lqrDq[5] = 0;
    for (let i = 0; i < NV; i++) lqrDx[i] = lqrDq[i];
    for (let i = 0; i < NV; i++) lqrDx[NV + i] = v[i] - lqrQvelRef[i];
    let s0 = EQ_CTRL[0]; for (let c = 0; c < N_STATE; c++) s0 -= lqrK[0 * N_STATE + c] * lqrDx[c];
    let s1 = EQ_CTRL[1]; for (let c = 0; c < N_STATE; c++) s1 -= lqrK[1 * N_STATE + c] * lqrDx[c];
    let s2 = EQ_CTRL[2]; for (let c = 0; c < N_STATE; c++) s2 -= lqrK[2 * N_STATE + c] * lqrDx[c];
    // Extra lateral damping on the pelvis_x servo helps when contact slip or WASM linearization
    // noise underestimates the roll channel. It uses the same tangent-state coordinates as the
    // LQR, so it vanishes exactly at the trim.
    s1 -= LQR_LAT_ROLL_K * lqrDx[3];
    s1 -= LQR_LAT_ROLLRATE_K * lqrDx[NV + 3];
    s1 -= LQR_LAT_PELVISX_K * lqrDx[8];
    s1 -= LQR_LAT_PELVISX_RATE_K * lqrDx[NV + 8];
    // nu order matches xml: [pelvis_y, pelvis_x, wheel] == ctrl[0,1,2] == A,B rows
    data.ctrl[ACT.pelvis_y] = Math.max(PELVIS_Y_LO, Math.min(PELVIS_Y_HI, s0));
    data.ctrl[ACT.pelvis_x] = Math.max(PELVIS_X_LO, Math.min(PELVIS_X_HI, s1));
    data.ctrl[ACT.wheel] = Math.max(-WHEEL_TORQUE_MAX, Math.min(WHEEL_TORQUE_MAX, s2));
  }

  function applyControls() {
    if (lqrEnabled) {
      applyLqr();
      return;
    }
    data.ctrl[ACT.wheel] = (keys.wheelFwd ? WHEEL_TORQUE : 0) - (keys.wheelBwd ? WHEEL_TORQUE : 0);
    const padY = -padNormY;
    const target_y = padY >= 0 ? padY * PELVIS_Y_HI : padY * (-PELVIS_Y_LO);
    // Horizontal pad ↔ pelvis_x was inverted relative to the on-pad "roll ← / roll →" cues.
    const rollNorm = -padNormX;
    const target_x = rollNorm >= 0 ? rollNorm * PELVIS_X_HI : rollNorm * (-PELVIS_X_LO);
    data.ctrl[ACT.pelvis_y] = target_y;
    data.ctrl[ACT.pelvis_x] = target_x;
  }

  renderer.domElement.style.touchAction = "none";

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.6, 0);
  controls.enableDamping = true;
  controls.enableRotate = true;
  controls.enablePan = true;
  controls.enableZoom = true;
  controls.minDistance = 0.5;
  controls.maxDistance = 15;
  // Standard mappings: left = orbit, middle = dolly, right = pan; one finger = orbit, two = pinch+pan.
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
  };
  controls.touches = {
    ONE: THREE.TOUCH?.ROTATE ?? 0,
    TWO: THREE.TOUCH?.DOLLY_PAN ?? 2,
  };

  const uiPanel = document.createElement("div");
  uiPanel.id = "unicycle-ui";
  uiPanel.style.cssText = [
    "position:absolute",
    "top:12px",
    "left:12px",
    "display:flex",
    "flex-direction:column",
    "gap:8px",
    "width:min(360px, calc(100% - 24px))",
    "max-height:calc(100% - 24px)",
    "padding:12px 14px",
    "border:1px solid rgba(255,255,255,0.14)",
    "border-radius:12px",
    "background:rgba(10,16,24,0.66)",
    "box-shadow:0 12px 28px rgba(0,0,0,0.28)",
    "backdrop-filter:blur(10px)",
    "color:#fff",
    "font:12px/1.5 sans-serif",
    "overflow-y:auto",
    "pointer-events:none",
    "text-shadow:0 1px 2px rgba(0,0,0,0.55)",
  ].join(";");
  uiPanel.innerHTML = [
    "<div style='font:600 14px/1.4 sans-serif'>Unicycle MuJoCo</div>",
    "<div style='color:rgba(255,255,255,0.78)'>Drag to orbit, right-drag to pan, wheel to zoom.</div>",
    "<div id='unicycle-manual-copy'>Manual controls: use the pad to lean and <kbd>W</kbd>/<kbd>S</kbd> or <kbd>↑</kbd>/<kbd>↓</kbd> for wheel torque.</div>",
    "<div><span style='color:#3dff8c;font-weight:600'>Goal:</span> reach the green beacon.</div>",
    "<div><kbd>Space</kbd> pause · <kbd>R</kbd> reset</div>",
    "<label style='display:flex;align-items:flex-start;gap:8px;pointer-events:auto;cursor:pointer'>",
    "  <input id='unicycle-lqr' type='checkbox' style='margin-top:2px'>",
    "  <span><b>LQR</b> upright balance only (<kbd>L</kbd>)</span>",
    "</label>",
    "<div id='unicycle-lqr-hud' style='color:#a0ffd0;font:11px/1.4 monospace;display:none'>LQR upright balance is active.</div>",
    "<label style='display:flex;align-items:flex-start;gap:8px;pointer-events:auto;cursor:pointer'>",
    "  <input id='unicycle-ragdoll' type='checkbox' style='margin-top:2px'>",
    "  <span>Show humanoid (<kbd>H</kbd>)</span>",
    "</label>",
  ].join("");
  for (const kbd of uiPanel.querySelectorAll("kbd")) {
    kbd.style.cssText = [
      "display:inline-block",
      "padding:1px 5px",
      "border:1px solid rgba(255,255,255,0.18)",
      "border-radius:5px",
      "background:rgba(255,255,255,0.08)",
      "font:11px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
      "color:#fff",
      "text-shadow:none",
    ].join(";");
  }
  container.appendChild(uiPanel);
  const lqrHudNode = uiPanel.querySelector("#unicycle-lqr-hud");

  function applyLqrTrimState() {
    mujoco.mj_resetData(model, data);
    for (let i = 0; i < NQ; i++) data.qpos[i] = EQ_QPOS[i];
    for (let i = 0; i < NV; i++) data.qvel[i] = 0;
    for (let i = 0; i < NU; i++) data.ctrl[i] = EQ_CTRL[i];
    mujoco.mj_forward(model, data);
  }

  function resetSimulation() {
    if (lqrEnabled) applyLqrTrimState();
    else {
      mujoco.mj_resetData(model, data);
      mujoco.mj_forward(model, data);
    }
    mujoco.mj_resetData(model_v, data_v);
    mujoco.mj_forward(model_v, data_v);
    if (ragdollEnabled) syncVisualFromSimplified();
    respawnTarget();
  }

  function setLqrEnabled(enabled) {
    lqrEnabled = !!enabled;
    const c = document.getElementById("unicycle-lqr");
    if (c) c.checked = lqrEnabled;
    if (lqrHudNode) lqrHudNode.style.display = lqrEnabled ? "block" : "none";
    if (lqrEnabled) resetPad();
    updateManualUi();
    if (lqrEnabled) {
      applyLqrTrimState();
      if (ragdollEnabled) syncVisualFromSimplified();
    }
  }

  function setRagdollEnabled(enabled) {
    ragdollEnabled = !!enabled;
    const cb = document.getElementById("unicycle-ragdoll");
    if (cb) cb.checked = ragdollEnabled;
    applyRagdollVisibility();
    if (ragdollEnabled) syncVisualFromSimplified();
  }
  const ragdollCheckbox = uiPanel.querySelector("#unicycle-ragdoll");
  if (ragdollCheckbox) {
    ragdollCheckbox.addEventListener("change", (e) => setRagdollEnabled(e.target.checked));
  }
  const lqrCheckbox = uiPanel.querySelector("#unicycle-lqr");
  if (lqrCheckbox) {
    lqrCheckbox.addEventListener("change", (e) => setLqrEnabled(e.target.checked));
  }

  // Ensure the pad can be positioned relative to the container.
  const containerPos = getComputedStyle(container).position;
  if (containerPos === "static" || !containerPos) {
    container.style.position = "relative";
  }

  const PAD_SIZE = 180;
  const pad = document.createElement("div");
  pad.id = "unicycle-pad";
  pad.style.cssText = [
    "position:absolute",
    "right:12px",
    "bottom:12px",
    `width:${PAD_SIZE}px`,
    `height:${PAD_SIZE}px`,
    "background:rgba(0,0,0,0.35)",
    "border:1px solid rgba(255,255,255,0.55)",
    "border-radius:6px",
    "box-shadow:0 1px 4px rgba(0,0,0,0.5)",
    "touch-action:none",
    "cursor:crosshair",
    "user-select:none",
  ].join(";");

  // Crosshair at the pad center (always visible so the user knows where neutral is).
  const crosshair = document.createElement("div");
  crosshair.style.cssText = [
    "position:absolute",
    "left:0",
    "top:50%",
    "width:100%",
    "height:0",
    "border-top:1px dashed rgba(255,255,255,0.35)",
    "pointer-events:none",
  ].join(";");
  pad.appendChild(crosshair);
  const crosshairV = document.createElement("div");
  crosshairV.style.cssText = [
    "position:absolute",
    "top:0",
    "left:50%",
    "width:0",
    "height:100%",
    "border-left:1px dashed rgba(255,255,255,0.35)",
    "pointer-events:none",
  ].join(";");
  pad.appendChild(crosshairV);

  const centerDot = document.createElement("div");
  const CENTER_DOT = 8;
  centerDot.style.cssText = [
    "position:absolute",
    `left:${(PAD_SIZE - CENTER_DOT) / 2}px`,
    `top:${(PAD_SIZE - CENTER_DOT) / 2}px`,
    `width:${CENTER_DOT}px`,
    `height:${CENTER_DOT}px`,
    "border:1px solid rgba(255,255,255,0.8)",
    "border-radius:50%",
    "pointer-events:none",
  ].join(";");
  pad.appendChild(centerDot);

  const PAD_LABEL_CSS =
    "position:absolute;font:10px/1 sans-serif;color:rgba(255,255,255,0.85);text-shadow:0 1px 2px #000;pointer-events:none;";
  const labels = [
    { text: "tilt fwd", css: "left:50%;top:4px;transform:translateX(-50%);" },
    { text: "tilt back", css: "left:50%;bottom:4px;transform:translateX(-50%);" },
    { text: "roll ←", css: "left:4px;top:50%;transform:translateY(-50%);" },
    { text: "roll →", css: "right:4px;top:50%;transform:translateY(-50%);" },
  ];
  for (const l of labels) {
    const el = document.createElement("div");
    el.textContent = l.text;
    el.style.cssText = PAD_LABEL_CSS + l.css;
    pad.appendChild(el);
  }

  const KNOB = 18;
  const knob = document.createElement("div");
  knob.style.cssText = [
    "position:absolute",
    `width:${KNOB}px`,
    `height:${KNOB}px`,
    `left:${(PAD_SIZE - KNOB) / 2}px`,
    `top:${(PAD_SIZE - KNOB) / 2}px`,
    "border-radius:50%",
    "background:rgba(120,200,255,0.9)",
    "border:1px solid rgba(255,255,255,0.9)",
    "box-shadow:0 0 6px rgba(120,200,255,0.6)",
    "pointer-events:none",
    "transition:background 0.1s",
  ].join(";");
  pad.appendChild(knob);

  container.appendChild(pad);

  const manualControlsNote = uiPanel.querySelector("#unicycle-manual-copy");
  function updateManualUi() {
    if (manualControlsNote) {
      manualControlsNote.innerHTML = lqrEnabled
        ? "Manual controls are paused while LQR is on."
        : "Manual controls: use the pad to lean and <kbd>W</kbd>/<kbd>S</kbd> or <kbd>↑</kbd>/<kbd>↓</kbd> for wheel torque.";
      for (const kbd of manualControlsNote.querySelectorAll("kbd")) {
        kbd.style.cssText = [
          "display:inline-block",
          "padding:1px 5px",
          "border:1px solid rgba(255,255,255,0.18)",
          "border-radius:5px",
          "background:rgba(255,255,255,0.08)",
          "font:11px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
          "color:#fff",
          "text-shadow:none",
        ].join(";");
      }
    }
    pad.style.opacity = lqrEnabled ? "0.45" : "1";
    pad.style.pointerEvents = lqrEnabled ? "none" : "auto";
  }

  function updatePadFromEvent(e) {
    const rect = pad.getBoundingClientRect();
    const half = rect.width / 2;
    const cx = rect.left + half;
    const cy = rect.top + rect.height / 2;
    let nx = (e.clientX - cx) / half;
    let ny = (cy - e.clientY) / half; // pad Y up is positive
    if (nx > 1) nx = 1; else if (nx < -1) nx = -1;
    if (ny > 1) ny = 1; else if (ny < -1) ny = -1;
    padNormX = nx;
    padNormY = ny;
    const kx = (PAD_SIZE - KNOB) / 2 + nx * (PAD_SIZE - KNOB) / 2;
    const ky = (PAD_SIZE - KNOB) / 2 - ny * (PAD_SIZE - KNOB) / 2;
    knob.style.left = `${kx}px`;
    knob.style.top = `${ky}px`;
  }

  function resetPad() {
    padActive = false;
    padNormX = 0;
    padNormY = 0;
    knob.style.left = `${(PAD_SIZE - KNOB) / 2}px`;
    knob.style.top = `${(PAD_SIZE - KNOB) / 2}px`;
    knob.style.background = "rgba(120,200,255,0.9)";
  }

  pad.addEventListener("pointerdown", (e) => {
    if (e.button !== undefined && e.button !== 0) return;
    padActive = true;
    knob.style.background = "rgba(255,190,90,0.95)";
    try { pad.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    updatePadFromEvent(e);
    e.preventDefault();
  });
  pad.addEventListener("pointermove", (e) => {
    if (!padActive) return;
    updatePadFromEvent(e);
    e.preventDefault();
  });
  const endPad = (e) => {
    if (!padActive) return;
    try { pad.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    resetPad();
    e.preventDefault();
  };
  pad.addEventListener("pointerup", endPad);
  pad.addEventListener("pointercancel", endPad);
  pad.addEventListener("pointerleave", (e) => {
    // If the browser didn't grant pointer capture, treat leaving as release.
    if (padActive && !pad.hasPointerCapture?.(e.pointerId)) resetPad();
  });

  const tmpPos = new THREE.Vector3();
  const tmpQuat = new THREE.Quaternion();
  const G = mujoco.mjtGeom || {};
  const planeType = G.mjGEOM_PLANE?.value ?? 0;

  // Build a Three.js Group per body by collecting renderable geoms (group < 3, skipping floor
  // for the visual model to avoid z-fighting with the simplified floor). Material/texture
  // lookup follows zalo/mujoco_wasm: per-geom `mat_rgba`, `mat_texid[mat*10 + mjTEXROLE_RGB]`,
  // `mat_texrepeat`, `mat_{specular,reflectance,shininess}`.
  function buildBodyGroups(m, { skipBodyNames = new Set() } = {}) {
    const groups = {};
    for (let g = 0; g < m.ngeom; g++) {
      if (m.geom_group[g] >= 3) continue;
      const bid = m.geom_bodyid[g];
      const bname = mujoco.mj_id2name(m, OBJ_BODY, bid) ?? "";
      const type = m.geom_type[g];
      const isPlane = type === planeType;
      if (m === model_v && isPlane) continue;
      if (skipBodyNames.has(bname)) continue;
      if (!(bid in groups)) {
        groups[bid] = new THREE.Group();
        scene.add(groups[bid]);
      }

      let color = [
        m.geom_rgba[g * 4],
        m.geom_rgba[g * 4 + 1],
        m.geom_rgba[g * 4 + 2],
        m.geom_rgba[g * 4 + 3],
      ];
      let map = null;
      let specular, reflectance, shininess;
      const matId = m.geom_matid ? m.geom_matid[g] : -1;
      if (matId !== -1) {
        color = [
          m.mat_rgba[matId * 4],
          m.mat_rgba[matId * 4 + 1],
          m.mat_rgba[matId * 4 + 2],
          m.mat_rgba[matId * 4 + 3],
        ];
        specular = m.mat_specular[matId];
        reflectance = m.mat_reflectance[matId];
        shininess = m.mat_shininess[matId];
        const texId = m.mat_texid[matId * MJ_NTEXROLE + MJ_TEXROLE_RGB];
        if (texId !== -1 && texId < modelTextures.length) {
          const entry = modelTextures[texId];
          // Skybox textures belong on the scene background, not on geoms.
          if (entry.type !== MJ_TEX_SKYBOX) {
            map = entry.tex.clone();
            map.needsUpdate = true;
            map.wrapS = THREE.RepeatWrapping;
            map.wrapT = THREE.RepeatWrapping;
            // Plane geoms (floors) ignore material texrepeat=1,1 and tile by world scale;
            // zalo/mujoco_wasm applies a hard-coded 50× repeat for that case. Do the same.
            if (isPlane) {
              map.repeat.set(50, 50);
            } else {
              map.repeat.set(
                m.mat_texrepeat[matId * 2] || 1,
                m.mat_texrepeat[matId * 2 + 1] || 1
              );
            }
          }
        }
      }

      const matParams = {
        color: new THREE.Color(color[0], color[1], color[2]),
        transparent: color[3] < 1,
        opacity: color[3],
        map,
      };
      if (matId !== -1) {
        matParams.roughness = 1.0 - shininess;
        matParams.metalness = 0.1;
        matParams.reflectivity = reflectance;
        matParams.specularIntensity = specular;
      } else {
        matParams.roughness = 0.55;
        matParams.metalness = 0.1;
      }
      const mat = new THREE.MeshPhysicalMaterial(matParams);
      const mesh = createMeshForGeom(mujoco, m, g, mat, isPlane);
      groups[bid].add(mesh);
    }
    return groups;
  }

  const bodies = buildBodyGroups(model);
  const bodies_v = buildBodyGroups(model_v);

  // Map simplified body-name → id for the ragdoll-toggle hide list.
  const simplifiedHideIds = new Set();
  for (const name of HIDE_ON_RAGDOLL) {
    const id = mujoco.mj_name2id(model, OBJ_BODY, name);
    if (id >= 0) simplifiedHideIds.add(id);
  }

  // Toggle state: when true, render the visual humanoid and hide overlapping simplified bodies.
  let ragdollEnabled = false;
  function applyRagdollVisibility() {
    for (const [bidStr, grp] of Object.entries(bodies)) {
      const bid = Number(bidStr);
      grp.visible = !ragdollEnabled || !simplifiedHideIds.has(bid);
    }
    for (const grp of Object.values(bodies_v)) {
      grp.visible = ragdollEnabled;
    }
  }
  applyRagdollVisibility();

  // One-way sync: simplified -> visual.
  //   * Free joint (wheel pose + twist): direct copy of qpos[0..6] / qvel[0..5].
  //     Both models have the wheel body as root, so these 7/6 values have identical
  //     meaning (position + quaternion / linear + angular velocity of the wheel in world).
  //   * Pelvis hinges + wheel revolute: pin angle and velocity by matching joint name;
  //     this drives the visual humanoid's upper body tilt from the simplified COM tilt
  //     and keeps the visual wheel spin in lock-step with the simplified wheel.
  //   * Abdomen triplet (z/y/x): held at zero so the humanoid's upper body stays rigid
  //     above the pelvis hinge, matching the simplified rigid com segment.
  // All other visual DOFs (legs, arms, pedals) evolve under gravity, joint stiffness,
  // and the foot<->pedal welds / hand<->head connect equalities each visual step.
  function writeVisualPins() {
    const qs = data.qpos;
    const vs = data.qvel;
    data_v.qpos[0] = qs[0]; data_v.qpos[1] = qs[1]; data_v.qpos[2] = qs[2];
    data_v.qpos[3] = qs[3]; data_v.qpos[4] = qs[4]; data_v.qpos[5] = qs[5]; data_v.qpos[6] = qs[6];
    data_v.qvel[0] = vs[0]; data_v.qvel[1] = vs[1]; data_v.qvel[2] = vs[2];
    data_v.qvel[3] = vs[3]; data_v.qvel[4] = vs[4]; data_v.qvel[5] = vs[5];
    for (const p of pinMap) {
      data_v.qpos[p.v.qposadr] = data.qpos[p.s.qposadr];
      data_v.qvel[p.v.dofadr] = data.qvel[p.s.dofadr];
    }
    // for (const a of zeroPinAddrs) {
    //   data_v.qpos[a.qposadr] = 0;
    //   data_v.qvel[a.dofadr] = 0;
    // }
  }
  function syncVisualFromSimplified() {
    writeVisualPins();
    for (let i = 0; i < model_v.nu; i++) data_v.ctrl[i] = 0;
    mujoco.mj_step(model_v, data_v);
    // Re-pin after step so pinned DOFs don't drift due to solver interactions.
    writeVisualPins();
    mujoco.mj_forward(model_v, data_v);
  }

  // Defaults after bodies + ragdoll state exist (avoid TDZ on ragdollEnabled).
  setRagdollEnabled(true);
  setLqrEnabled(true);

  let paused = false;
  let lastTime = performance.now();

  function onKeyDown(e) {
    switch (e.code) {
      case "KeyL":
        if (!e.repeat) setLqrEnabled(!lqrEnabled);
        e.preventDefault();
        return;
      case "KeyW":
        if (!lqrEnabled) keys.wheelFwd = true;
        break;
      case "KeyS":
        if (!lqrEnabled) keys.wheelBwd = true;
        break;
      case "ArrowUp":
        if (!lqrEnabled) keys.wheelFwd = true;
        break;
      case "ArrowDown":
        if (!lqrEnabled) keys.wheelBwd = true;
        break;
      case "ArrowLeft":
        break;
      case "ArrowRight":
        break;
      case "Space": paused = !paused; e.preventDefault(); return;
      case "KeyR":
        resetSimulation();
        e.preventDefault();
        return;
      case "KeyH":
        setRagdollEnabled(!ragdollEnabled);
        e.preventDefault();
        return;
      default: return;
    }
    e.preventDefault();
  }

  function onKeyUp(e) {
    switch (e.code) {
      case "KeyW":
        keys.wheelFwd = false;
        break;
      case "KeyS":
        keys.wheelBwd = false;
        break;
      case "ArrowUp":
        if (!lqrEnabled) keys.wheelFwd = false;
        break;
      case "ArrowDown":
        if (!lqrEnabled) keys.wheelBwd = false;
        break;
      case "ArrowLeft":
      case "ArrowRight":
        break;
      default: return;
    }
    e.preventDefault();
  }

  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  window.addEventListener("blur", () => {
    keys.wheelFwd = false;
    keys.wheelBwd = false;
  });

  function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const dt = (now - lastTime) / 1000;
    lastTime = now;

    if (!paused) {
      applyControls();
      const timestep = model.opt.timestep;
      let steps = Math.max(1, Math.floor(dt / timestep));
      steps = Math.min(steps, 5);
      for (let i = 0; i < steps; i++) {
        mujoco.mj_step(model, data);
        if (ragdollEnabled) syncVisualFromSimplified();
      }
    }

    // Advance the beam shader time and gently rotate the ring for a parallax cue.
    beamUniforms.uTime.value = now / 1000;
    glowRing.rotation.z += dt * 0.6;

    // Reached target? Check distance in the world XZ plane between the wheel and the
    // beacon using Three coords. xpos is MuJoCo (x, y, z) with z up, and we map to
    // Three.js as (x, z, -y) elsewhere — so the ground-plane coords are (mx, -my).
    const wx = data.xpos[S_WHEEL_BID * 3];
    const wy = data.xpos[S_WHEEL_BID * 3 + 1];
    const tx = targetGroup.position.x;
    const tz = targetGroup.position.z;
    const dx = wx - tx;
    const dz = -wy - tz;
    if (dx * dx + dz * dz < TARGET_REACH_R * TARGET_REACH_R) {
      respawnTarget();
    }

    for (let b = 0; b < model.nbody; b++) {
      if (bodies[b]) {
        getMujocoPos(data, b, tmpPos);
        getMujocoQuat(data, b, tmpQuat);
        bodies[b].position.copy(tmpPos);
        bodies[b].quaternion.copy(tmpQuat);
        bodies[b].updateMatrixWorld(true);
      }
    }
    if (ragdollEnabled) {
      for (let b = 0; b < model_v.nbody; b++) {
        if (bodies_v[b]) {
          getMujocoPos(data_v, b, tmpPos);
          getMujocoQuat(data_v, b, tmpQuat);
          bodies_v[b].position.copy(tmpPos);
          bodies_v[b].quaternion.copy(tmpQuat);
          bodies_v[b].updateMatrixWorld(true);
        }
      }
    }

    controls.update();
    renderer.render(scene, camera);
  }

  window.unicyclePause = () => { paused = !paused; };
  // Debug helper for verifying wheel alignment between models.
  window.unicycleDumpWheels = () => {
    const s_wheel = mujoco.mj_name2id(model, OBJ_BODY, "wheel");
    const v_wheel = mujoco.mj_name2id(model_v, OBJ_BODY, "wheel_and_crank");
    const s_seat = mujoco.mj_name2id(model, OBJ_BODY, "seat");
    const v_seat = mujoco.mj_name2id(model_v, OBJ_BODY, "seat");
    const sp = (d, b) => [d.xpos[b * 3], d.xpos[b * 3 + 1], d.xpos[b * 3 + 2]];
    const fmt = (a) => `[${a.map((x) => x.toFixed(4)).join(", ")}]`;
    console.log("simp wheel", fmt(sp(data, s_wheel)), "full wheel", fmt(sp(data_v, v_wheel)));
    console.log("simp seat ", fmt(sp(data, s_seat)),  "full seat ", fmt(sp(data_v, v_seat)));
  };
  window.unicycleReset = () => {
    resetSimulation();
  };
  window.unicycleToggleRagdoll = () => setRagdollEnabled(!ragdollEnabled);
  window.unicycleToggleLqr = () => setLqrEnabled(!lqrEnabled);
  window.unicycleRespawnTarget = respawnTarget;

  animate();

  window.addEventListener("resize", () => {
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}

main().catch((err) => {
  console.error("Unicycle init failed:", err);
});

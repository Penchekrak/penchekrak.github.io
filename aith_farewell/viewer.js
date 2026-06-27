import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

const TRAJECTORY_LINE_WIDTH = 2.5;
const STRESSED_TRAJECTORY_LINE_WIDTH = 14;
const DEFAULT_POINT_SIZE = 0.018;

const canvas = document.querySelector("#scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x111418, 1);

const scene = new THREE.Scene();
scene.add(new THREE.GridHelper(8, 32, 0x39434d, 0x252c33));
scene.add(new THREE.AxesHelper(0.6));

const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
camera.position.set(1.6, 1.2, 2.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

const ui = {
  counts: document.querySelector("#counts"),
  pose: document.querySelector("#pose"),
  stride: document.querySelector("#stride"),
  trajectoryStress: document.querySelector("#trajectoryStress"),
  reset: document.querySelector("#reset"),
  poseLabel: document.querySelector("#poseLabel"),
  thumb: document.querySelector("#thumb")
};

let manifest;
let poses;
let pointGeometry;
let floorMesh;
let frustumLines;
let activeFrustum;
let trajectoryLine;
let trajectoryMaterial;

init();

async function init() {
  [manifest, poses] = await Promise.all([
    fetchJson("./assets/manifest.json"),
    fetchJson("./assets/poses.json")
  ]);
  const triangles = manifest.counts.mesh_triangles || 0;
  const meshText = triangles ? ` / ${triangles.toLocaleString()} tris` : "";
  ui.counts.textContent = `${manifest.counts.points.toLocaleString()} points${meshText} / ${manifest.counts.poses} poses`;
  ui.pose.max = Math.max(0, poses.poses.length - 1);
  const pointsBuffer = await fetchArrayBuffer("./assets/points.bin");
  addPointCloud(pointsBuffer);
  await addFloorMeshIfAvailable();
  addTrajectory();
  rebuildFrustums();
  updateActivePose(0);
  frameScene();
  bindControls();
  animate();
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}`);
  return response.json();
}

async function fetchArrayBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}`);
  return response.arrayBuffer();
}

function addPointCloud(buffer) {
  const bytesPerPoint = 16;
  const count = Math.floor(buffer.byteLength / bytesPerPoint);
  const view = new DataView(buffer);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const offset = index * bytesPerPoint;
    positions[index * 3 + 0] = view.getFloat32(offset + 0, true);
    positions[index * 3 + 1] = view.getFloat32(offset + 4, true);
    positions[index * 3 + 2] = view.getFloat32(offset + 8, true);
    colors[index * 3 + 0] = view.getUint8(offset + 12) / 255;
    colors[index * 3 + 1] = view.getUint8(offset + 13) / 255;
    colors[index * 3 + 2] = view.getUint8(offset + 14) / 255;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  pointGeometry = geometry;
  const material = new THREE.PointsMaterial({
    size: DEFAULT_POINT_SIZE,
    vertexColors: true,
    sizeAttenuation: true
  });
  scene.add(new THREE.Points(geometry, material));
}

async function addFloorMeshIfAvailable() {
  if (!manifest.mesh?.index_asset || !manifest.mesh.triangle_count || !pointGeometry) return;
  const indexBuffer = await fetchArrayBuffer(`./${manifest.mesh.index_asset}`);
  const indices = new Uint32Array(indexBuffer);
  const geometry = pointGeometry.clone();
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  const material = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.82,
    depthWrite: false
  });
  floorMesh = new THREE.Mesh(geometry, material);
  floorMesh.renderOrder = -1;
  scene.add(floorMesh);
}

function addTrajectory() {
  const positions = [];
  for (const pose of poses.poses) {
    const matrix = poseMatrix(pose);
    const position = new THREE.Vector3().setFromMatrixPosition(matrix);
    positions.push(position.x, position.y, position.z);
  }
  const geometry = new LineGeometry();
  geometry.setPositions(positions);
  trajectoryMaterial = new LineMaterial({
    color: 0x62c4ff,
    linewidth: trajectoryLineWidth(),
    transparent: true,
    opacity: 0.96
  });
  trajectoryLine = new Line2(geometry, trajectoryMaterial);
  trajectoryLine.computeLineDistances();
  scene.add(trajectoryLine);
}

function trajectoryLineWidth() {
  return ui.trajectoryStress.checked ? STRESSED_TRAJECTORY_LINE_WIDTH : TRAJECTORY_LINE_WIDTH;
}

function rebuildFrustums() {
  if (frustumLines) scene.remove(frustumLines);
  const stride = Math.max(1, Number(ui.stride.value));
  const positions = [];
  for (let index = 0; index < poses.poses.length; index += stride) {
    positions.push(...frustumSegments(poses.poses[index]));
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  frustumLines = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color: 0x8aa0ad, transparent: true, opacity: 0.38 })
  );
  scene.add(frustumLines);
}

function updateActivePose(index) {
  const pose = poses.poses[index];
  if (!pose) return;
  if (activeFrustum) scene.remove(activeFrustum);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(frustumSegments(pose), 3));
  activeFrustum = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color: 0xffd166 })
  );
  scene.add(activeFrustum);
  ui.poseLabel.textContent = `frame ${pose.frame_index} / ${pose.timestamp_s.toFixed(2)}s`;
  ui.thumb.src = pose.thumbnail || "";
}

function frustumSegments(pose) {
  const matrix = poseMatrix(pose);
  const f = pose.frustum;
  const depth = f.depth_m || 0.18;
  const corners = [
    imageCornerToCamera(0, 0, f, depth),
    imageCornerToCamera(f.image_width, 0, f, depth),
    imageCornerToCamera(f.image_width, f.image_height, f, depth),
    imageCornerToCamera(0, f.image_height, f, depth)
  ].map((point) => point.applyMatrix4(matrix));
  const origin = new THREE.Vector3().setFromMatrixPosition(matrix);
  const lines = [];
  for (const corner of corners) pushSegment(lines, origin, corner);
  pushSegment(lines, corners[0], corners[1]);
  pushSegment(lines, corners[1], corners[2]);
  pushSegment(lines, corners[2], corners[3]);
  pushSegment(lines, corners[3], corners[0]);
  return lines;
}

function imageCornerToCamera(u, v, f, depth) {
  return new THREE.Vector3(
    ((u - f.cx) / f.fx) * depth,
    -((v - f.cy) / f.fy) * depth,
    -depth
  );
}

function pushSegment(target, a, b) {
  target.push(a.x, a.y, a.z, b.x, b.y, b.z);
}

function poseMatrix(pose) {
  return new THREE.Matrix4().set(...pose.matrix);
}

function bindControls() {
  ui.pose.addEventListener("input", () => updateActivePose(Number(ui.pose.value)));
  ui.stride.addEventListener("input", rebuildFrustums);
  ui.trajectoryStress.addEventListener("change", () => {
    if (trajectoryMaterial) trajectoryMaterial.linewidth = trajectoryLineWidth();
  });
  ui.reset.addEventListener("click", frameScene);
  window.addEventListener("resize", resize);
}

function frameScene() {
  const bounds = manifest.bounds || { min: [-1, -1, -1], max: [1, 1, 1] };
  const min = new THREE.Vector3(...bounds.min);
  const max = new THREE.Vector3(...bounds.max);
  const box = new THREE.Box3(min, max);
  if (trajectoryLine) box.expandByObject(trajectoryLine);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const radius = Math.max(size.length() * 0.55, 0.8);
  camera.position.copy(center).add(new THREE.Vector3(radius, radius * 0.7, radius * 1.2));
  controls.target.copy(center);
  controls.update();
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
  if (trajectoryMaterial) trajectoryMaterial.resolution.set(width, height);
}

function animate() {
  resize();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

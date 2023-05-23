import * as THREE from 'three';


import {GUI} from 'three/addons/libs/lil-gui.module.min.js';

import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {DecalGeometry} from 'three/addons/geometries/DecalGeometry.js';

const container = document.getElementById('container');
THREE.Cache.enabled = true;
let renderer, scene, camera;
let mesh;
let raycaster;
let line;
let decalTexture;
let controls;

const intersection = {
    intersects: false,
    point: new THREE.Vector3(),
    normal: new THREE.Vector3()
};
const mouse = new THREE.Vector2();
const intersects = [];

const textureLoader = new THREE.TextureLoader();

const decals = [];
const orientations = [];
const positions = [];
const scales = [];
let mouseHelper;
const position = new THREE.Vector3();
const orientation = new THREE.Euler();

const params = {
    'image url': "https://avatars.githubusercontent.com/u/13419363?s=280&v=4",
    'rotate last': 0.0,
    'scale last': 0.0,
    'delete last': function () {

        removeLastDecal();

    },
    'reset': function () {

        removeDecals();

    }
};

init();
animate();

function init() {

    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
    camera.position.z = 120;

    controls = new OrbitControls(camera, renderer.domElement);
    controls.minDistance = 50;
    controls.maxDistance = 200;

    scene.add(new THREE.AmbientLight(0xffffff));

    const geometry = new THREE.BufferGeometry();
    geometry.setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);

    line = new THREE.Line(geometry, new THREE.LineBasicMaterial());
    scene.add(line);

    loadArc();

    raycaster = new THREE.Raycaster();

    mouseHelper = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 10), new THREE.MeshNormalMaterial());
    mouseHelper.visible = false;
    scene.add(mouseHelper);

    window.addEventListener('resize', onWindowResize);

    let moved = false;

    controls.addEventListener('change', function () {

        moved = true;

    });

    window.addEventListener('pointerdown', function () {

        moved = false;

    });

    window.addEventListener('pointerup', function (event) {

        if (moved === false) {

            checkIntersection(event.clientX, event.clientY);

            if (intersection.intersects) shoot();

        }

    });

    window.addEventListener('pointermove', onPointerMove);

    function onPointerMove(event) {

        if (event.isPrimary) {

            checkIntersection(event.clientX, event.clientY);

        }

    }

    function checkIntersection(x, y) {

        if (mesh === undefined) return;

        mouse.x = (x / window.innerWidth) * 2 - 1;
        mouse.y = -(y / window.innerHeight) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        raycaster.intersectObject(mesh, false, intersects);

        if (intersects.length > 0) {

            const p = intersects[0].point;
            mouseHelper.position.copy(p);
            intersection.point.copy(p);

            const n = intersects[0].face.normal.clone();
            n.transformDirection(mesh.matrixWorld);
            n.multiplyScalar(10);
            n.add(intersects[0].point);

            intersection.normal.copy(intersects[0].face.normal);
            mouseHelper.lookAt(n);

            const positions = line.geometry.attributes.position;
            positions.setXYZ(0, p.x, p.y, p.z);
            positions.setXYZ(1, n.x, n.y, n.z);
            positions.needsUpdate = true;

            intersection.intersects = true;

            intersects.length = 0;

        } else {

            intersection.intersects = false;

        }

    }

    const gui = new GUI();

    gui.add(params, 'image url');
    let r = gui.add(params, 'rotate last', -180, 180);
    let s = gui.add(params, 'scale last', -1, 2);
    r.$input.addEventListener('input', rotateDecal);
    r.$slider.addEventListener('mouseup', rotateDecal);
    s.$input.addEventListener('input', scaleDecal);
    s.$slider.addEventListener('mouseup', scaleDecal);
    gui.add(params, 'delete last');
    gui.add(params, 'reset');
    gui.open();

}

function rotateDecal() {
    if ( decals.length > 0 ) {
        scene.remove(decals.pop());
        orientation.copy(orientations.pop());
        const scale = scales.at(-1);
        orientation.z = params['rotate last'] / 180.0 * Math.PI;
        const new_m = new THREE.Mesh(
            new DecalGeometry(mesh, positions.at(-1), orientation, new THREE.Vector3(scale, scale, scale)),
            new THREE.MeshPhongMaterial({
                map: decalTexture,
                normalScale: new THREE.Vector2(1, 1),
                shininess: 0,
                transparent: true,
                opacity: 0.5,
                depthTest: true,
                depthWrite: false,
                polygonOffset: true,
                polygonOffsetFactor: -4,
                wireframe: false
            })
        );
        decals.push(new_m);
        orientations.push(orientation)
        scene.add(new_m);
    }
}

function scaleDecal() {
    if ( decals.length > 0 ) {
        scene.remove(decals.pop());
        scales.pop();
        const scale = 30 * 10 ** params['scale last']
        const new_m = new THREE.Mesh(
            new DecalGeometry(mesh, positions.at(-1), orientations.at(-1), new THREE.Vector3(scale, scale, scale)),
            new THREE.MeshPhongMaterial({
                map: decalTexture,
                normalScale: new THREE.Vector2(1, 1),
                shininess: 0,
                transparent: true,
                opacity: 0.5,
                depthTest: true,
                depthWrite: false,
                polygonOffset: true,
                polygonOffsetFactor: -4,
                wireframe: false
            })
        );
        decals.push(new_m);
        scales.push(scale);
        scene.add(new_m);
    }
}

function loadArc() {

    const loader = new GLTFLoader();

    loader.load('arc_model/arc_small_lower_quadri_sep.gltf', function (gltf) {
        mesh = gltf.scene.children[0];
        scene.add(mesh);
        mesh.scale.set(10, 10, 10);
        mesh.position.y = mesh.position.y - 10;
    });

}

function shoot() {
    decalTexture = textureLoader.load(params['image url']);
    decalTexture.colorSpace = THREE.SRGBColorSpace;
    position.copy(intersection.point);
    orientation.copy(mouseHelper.rotation);
    const scale = 30
    const m = new THREE.Mesh(
        new DecalGeometry(mesh, position, orientation, new THREE.Vector3(scale, scale, scale)),
        new THREE.MeshPhongMaterial({
            map: decalTexture,
            normalScale: new THREE.Vector2(1, 1),
            shininess: 0,
            transparent: true,
            opacity: 0.5,
            depthTest: true,
            depthWrite: false,
            polygonOffset: true,
            polygonOffsetFactor: -4,
            wireframe: false
        })
    );

    decals.push(m);
    orientations.push(orientation);
    positions.push(position);
    scales.push(scale);
    scene.add(m);
}

function removeLastDecal() {

    scene.remove(decals.pop())

}

function removeDecals() {

    decals.forEach(function (d) {

        scene.remove(d);

    });

    decals.length = 0;
    orientations.length = 0;
    positions.length = 0;

}

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);

}

function animate() {

    requestAnimationFrame(animate);

    renderer.render(scene, camera);

}

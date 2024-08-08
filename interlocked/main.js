import * as THREE from 'three';

import {GUI} from 'three/addons/libs/lil-gui.module.min.js';

import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

let camera, scene, renderer;
let cameraControls;
let effectController;
let ambientLight, light;

let N, M, M_2, R, parametrization, alpha, alpha_2;
const pointmaterial = new THREE.PointsMaterial({size: 0.1, vertexColors: false});
const linematerial = new THREE.LineBasicMaterial({
    linewidth: 3,
    color: 0x0000ff
});
const redlinematerial = new THREE.LineBasicMaterial({
    color: 0xff0000
});
const raycaster = new THREE.Raycaster();

let hexahedra_collection = new THREE.Group();
const materials = {};
const box_indices = [
    2, 1, 0,
    0, 3, 2,
    0, 4, 7,
    7, 3, 0,
    0, 1, 5,
    5, 4, 0,
    1, 2, 6,
    6, 5, 1,
    2, 3, 7,
    7, 6, 2,
    4, 5, 6,
    6, 7, 4
];

const cube_vertices_from_hex_edges = [
    [0, 2, 4],
    [0, 1, 2],
    [0, 1, 5],
    [0, 4, 5],
    [3, 4, 2],
    [1, 2, 3],
    [1, 3, 5],
    [3, 4, 5],
]

init();
render();

// function onPointerDown(event) {
//     // console.log('pointer down');
//     let mouse = new THREE.Vector2();
//     mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
//     mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
//
//     raycaster.setFromCamera(mouse, camera);
//     const intersects = raycaster.intersectObjects(scene.children, false);
//     // console.log(intersects);
//     if (intersects.length > 0) {
//         const object = intersects[0].object;
//         console.log(object.userIndex)
//     }
//
// }

function init() {

    const container = document.createElement('div');
    document.body.appendChild(container);

    const canvasWidth = window.innerWidth;
    const canvasHeight = window.innerHeight;

    // CAMERA
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 80000);
    camera.position.set(-6, 5.5, 13);

    // LIGHTS
    ambientLight = new THREE.AmbientLight(0x7c7c7c, 3.0);

    light = new THREE.DirectionalLight(0xFFFFFF, 3.0);
    light.position.set(0.32, 0.39, 0.7);
    let light2 = new THREE.DirectionalLight(0xFFFFFF, 3.0);
    light2.position.set(-0.32, -0.39, -0.7);
    // RENDERER
    renderer = new THREE.WebGLRenderer({antialias: true});
    // renderer = new SVGRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(canvasWidth, canvasHeight);
    // renderer.shadowMap.enabled = true;

    container.appendChild(renderer.domElement);

    // EVENTS
    window.addEventListener('resize', onWindowResize);

    // CONTROLS
    cameraControls = new OrbitControls(camera, renderer.domElement);
    cameraControls.addEventListener('change', render);


    materials['wireframe'] = new THREE.MeshBasicMaterial({wireframe: true});
    materials['flat'] = new THREE.MeshPhongMaterial({specular: 0x000000, flatShading: true, side: THREE.DoubleSide});

    // scene itself
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xAAAAAA);

    scene.add(ambientLight);
    scene.add(light);
    scene.add(light2);
    // window.addEventListener('pointerdown', onPointerDown);


    // GUI
    setupGui();

}

// EVENT HANDLERS

function onWindowResize() {

    const canvasWidth = window.innerWidth;
    const canvasHeight = window.innerHeight;

    renderer.setSize(canvasWidth, canvasHeight);

    camera.aspect = canvasWidth / canvasHeight;
    camera.updateProjectionMatrix();

    render();

}

function setupGui() {

    effectController = {
        N: 15,
        M: 10,
        R: 2 * Math.sqrt(3),
        parametrization: 'naive',
        alpha: 1.0
    };

    const gui = new GUI();
    gui.add(effectController, 'N', [10, 15, 20, 25, 30, 35, 40, 45, 50]).name('N').onChange(render);
    gui.add(effectController, 'M', [10, 15, 20, 25, 30, 35, 40, 45, 50]).name('M').onChange(render);
    gui.add(effectController, 'R', 1.0, 10).name('outer ring radius').onChange(render);
    gui.add(effectController, 'parametrization', ['naive', 'approximately conformal', 'locally conformal']).name('parametrization').onChange(render);
    gui.add(effectController, 'alpha', 0.0, 2.0).name('alpha').onChange(render);
}


//

function render() {

    if (effectController.N !== N ||
        effectController.M !== M_2 ||
        effectController.R !== R ||
        effectController.parametrization !== parametrization ||
        effectController.alpha !== alpha_2) {

        N = effectController.N;
        M_2 = effectController.M;
        M = 2 * effectController.M;
        R = effectController.R;
        parametrization = effectController.parametrization;
        alpha = Math.asin(Math.sqrt(3) / 3) * effectController.alpha;
        alpha_2 = effectController.alpha;

        rebuild();

    }

    renderer.render(scene, camera);

}

function liftToTorus(planarVertex) {
    switch (parametrization) {
        case 'naive':
            return new THREE.Vector3(
                Math.cos(2 * Math.PI * planarVertex.x) * (R + Math.cos(2 * Math.PI * planarVertex.y)),
                Math.sin(2 * Math.PI * planarVertex.x) * (R + Math.cos(2 * Math.PI * planarVertex.y)),
                Math.sin(2 * Math.PI * planarVertex.y)
            );

        case 'approximately conformal':
            return new THREE.Vector3(
                Math.cos(2 * Math.PI * planarVertex.x) * (R + Math.cos(2 * Math.PI * planarVertex.y + Math.sin(2 * Math.PI * planarVertex.y) / R)),
                Math.sin(2 * Math.PI * planarVertex.x) * (R + Math.cos(2 * Math.PI * planarVertex.y + Math.sin(2 * Math.PI * planarVertex.y) / R)),
                Math.sin(2 * Math.PI * planarVertex.y + Math.sin(2 * Math.PI * planarVertex.y) / R)
            );
        case 'locally conformal':
            let scale = Math.sqrt(R * R - 1 + 1e-8);
            let multiplier = (R * R - 1 + 1e-8) / (R - Math.cos(Math.PI * (2 * planarVertex.y - 1)) + 1e-8)
            return new THREE.Vector3(
                Math.cos(2 * Math.PI * planarVertex.x),
                Math.sin(2 * Math.PI * planarVertex.x),
                Math.sin(Math.PI * (2 * planarVertex.y - 1)) / scale
            ).multiplyScalar(multiplier);
    }
}

function normalAtTorus(planarVertex) {
    switch (parametrization) {
        case 'naive':
            return new THREE.Vector3(
                4 * Math.PI * Math.PI * (R + Math.cos(2 * Math.PI * planarVertex.y)) * Math.cos(2 * Math.PI * planarVertex.x) * Math.cos(2 * Math.PI * planarVertex.y),
                4 * Math.PI * Math.PI * (R + Math.cos(2 * Math.PI * planarVertex.y)) * Math.sin(2 * Math.PI * planarVertex.x) * Math.cos(2 * Math.PI * planarVertex.y),
                4 * Math.PI * Math.PI * (R + Math.cos(2 * Math.PI * planarVertex.y)) * Math.sin(2 * Math.PI * planarVertex.y)
            ).normalize();

        case 'approximately conformal':
            return new THREE.Vector3(
                4 * Math.PI * Math.PI * (R + Math.cos(2 * Math.PI * planarVertex.y)) * (R + Math.cos(2 * Math.PI * planarVertex.y + Math.sin(2 * Math.PI * planarVertex.y) / R)) * Math.cos(2 * Math.PI * planarVertex.x) * Math.cos(2 * Math.PI * planarVertex.y + Math.sin(2 * Math.PI * planarVertex.y) / R) / R,
                4 * Math.PI * Math.PI * (R + Math.cos(2 * Math.PI * planarVertex.y)) * (R + Math.cos(2 * Math.PI * planarVertex.y + Math.sin(2 * Math.PI * planarVertex.y) / R)) * Math.sin(2 * Math.PI * planarVertex.x) * Math.cos(2 * Math.PI * planarVertex.y + Math.sin(2 * Math.PI * planarVertex.y) / R) / R,
                4 * Math.PI * Math.PI * (R + Math.cos(2 * Math.PI * planarVertex.y)) * (R + Math.cos(2 * Math.PI * planarVertex.y + Math.sin(2 * Math.PI * planarVertex.y) / R)) * Math.sin(2 * Math.PI * planarVertex.y + Math.sin(2 * Math.PI * planarVertex.y) / R) / R
            ).normalize();
        case 'locally conformal':
            return new THREE.Vector3(
                4 * Math.PI ** 2 * (R ** 2 - 1) ** (3 / 2) * (-2 * R * Math.cos(Math.PI * planarVertex.y) ** 2 + R - 1) * Math.cos(2 * Math.PI * planarVertex.x) / (R + Math.cos(2 * Math.PI * planarVertex.y)) ** 3,
                4 * Math.PI ** 2 * (R ** 2 - 1) ** (3 / 2) * (-2 * R * Math.cos(Math.PI * planarVertex.y) ** 2 + R - 1) * Math.sin(2 * Math.PI * planarVertex.x) / (R + Math.cos(2 * Math.PI * planarVertex.y)) ** 3,
                -4 * Math.PI ** 2 * (R ** 2 - 1) ** 2 * Math.sin(2 * Math.PI * planarVertex.y) / (R + Math.cos(2 * Math.PI * planarVertex.y)) ** 3
            ).normalize();
    }
}


function rebuild() {

    if (hexahedra_collection !== undefined) {
        scene.remove(hexahedra_collection);
        hexahedra_collection.children.forEach((elem) => {
            elem.geometry.dispose();
        });
        for (let i = hexahedra_collection.children.length - 1; i >= 0; --i)
            hexahedra_collection.remove(hexahedra_collection.children[i]);
    }
    let hexahedra_array = new Array(N);
    let hexahedra_normals = new Array(N);
    let scaling_vector = new THREE.Vector2(3 * N, M * Math.sqrt(3) / 2);
    for (let n = 0; n < N; ++n) {
        hexahedra_array[n] = new Array(M);
        hexahedra_normals[n] = new Array(M);
        for (let m = 0; m < M; ++m) {
            let planar_vertices = new Array(6);
            for (let k = 0; k < 6; ++k) {
                planar_vertices[k] = new THREE.Vector2(Math.cos(2 * Math.PI * k / 6), Math.sin(2 * Math.PI * k / 6));
            }
            let center = new THREE.Vector2(3 * n + 3 * ((-1) ** m + 1) / 4, Math.sqrt(3) / 2 * m);
            for (let k = 0; k < 6; ++k) {
                planar_vertices[k].add(center).divide(scaling_vector);
            }
            // console.log(planar_vertices);
            let torus_vertices = new Array(6);
            for (let k = 0; k < 6; ++k) {
                torus_vertices[k] = liftToTorus(planar_vertices[k]);
            }
            // console.log(torus_vertices);
            hexahedra_array[n][m] = torus_vertices;
            let normals = [];
            for (let i = 0; i < 6; ++i) {
                normals.push(normalAtTorus(new THREE.Vector2().addVectors(planar_vertices[i], planar_vertices[(i + 1) % 6]).divideScalar(2.0)));
            }
            hexahedra_normals[n][m] = normals;
        }
    }

    // let modulus_N = function (i) {
    //     return (i % N + N) % N
    // };
    // let modulus_M = function (i) {
    //     return (i % M + M) % M
    // };
    // let hex_center = function (array) {
    //     // console.log(array, array.reduce((partialSum, a) => partialSum.add(a), new THREE.Vector3()))
    //     return array.reduce((partialSum, a) => partialSum.add(a), new THREE.Vector3()).divideScalar(6)
    // };

    let rotate_span = function (vec1, vec2) {
        return new THREE.Vector3().add(vec2).multiplyScalar(Math.cos(alpha)).sub(vec1.multiplyScalar(Math.sin(alpha)));
    }

    let solve = function (vec1, vec2, vec3, num1, num2, num3) {
        let b = new THREE.Vector3(num1, num2, num3);
        let A_inv = (new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeBasis(vec1, vec2, vec3))).transpose().invert();
        return b.applyMatrix3(A_inv);
    }

    for (let n = 0; n < N; ++n) {
        for (let m = 0; m < M; ++m) {
            let hex = hexahedra_array[n][m];
            let normals = hexahedra_normals[n][m];

            // let compute_normal = function (i, j) {
            //     let midpoint = new THREE.Vector3().add(hex[i]).add(hex[j]).divideScalar(2.0);
            //     hexahedra_collection.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([midpoint, new THREE.Vector3().add(midpoint).add(normals[i])]), linematerial));
            // };
            //
            //
            // // let normals = [];
            // for (let i = 0; i < 6; ++i) {
            //     compute_normal(i, (i + 1) % 6);
            // }

            let compute_cross = function (i, j) {
                if (i % 2 === 0) {
                    return new THREE.Vector3().add(hex[j]).sub(hex[i]).cross(normals[i]).normalize()
                }
                return new THREE.Vector3().add(hex[i]).sub(hex[j]).cross(normals[i]).normalize()
            }
            let crosses = [];
            for (let i = 0; i < 6; ++i) {
                crosses.push(compute_cross(i, (i + 1) % 6));
            }

            for (let i = 0; i < 6; ++i) {
                crosses[i] = rotate_span(normals[i], crosses[i]);
            }

            let compute_b = function (i, j) {
                return new THREE.Vector3().addVectors(hex[i], hex[j]).dot(crosses[i]) / 2;
            }
            let bs = [];
            for (let i = 0; i < 6; ++i) {
                bs.push(compute_b(i, (i + 1) % 6));
            }

            let vertices = [];
            for (let i = 0; i < 8; ++i) {
                let indices = cube_vertices_from_hex_edges[i];
                let solution = solve(
                    crosses[indices[0]],
                    crosses[indices[1]],
                    crosses[indices[2]],
                    bs[indices[0]],
                    bs[indices[1]],
                    bs[indices[2]],
                );
                vertices.push(solution)
            }
            let geometry = new THREE.BufferGeometry().setFromPoints(vertices);
            geometry.setIndex(box_indices);
            let mesh = new THREE.Mesh(geometry, materials['flat'])
            hexahedra_collection.add(mesh);
            geometry = new THREE.BufferGeometry().setFromPoints([vertices[1], vertices[5], vertices[4], vertices[7], vertices[3], vertices[2]]).scale(1.0001, 1.0001, 1.0001);
            hexahedra_collection.add(new THREE.LineLoop(geometry, linematerial));
        }
    }
    // let positions = new Float32Array(3 * 6 * N * M);
    // for (let n = 0; n < N; ++n) {
    //     for (let m = 0; m < M; ++m) {
    //         for (let k = 0; k < 6; ++k) {
    //             positions[((n * M + m) * 6 + k) * 3] = hexahedra_array[n][m][k].x;
    //             positions[((n * M + m) * 6 + k) * 3 + 1] = hexahedra_array[n][m][k].y;
    //             positions[((n * M + m) * 6 + k) * 3 + 2] = hexahedra_array[n][m][k].z;
    //         }
    //     }
    // }
    // let geometry = new THREE.BufferGeometry();
    // geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    // geometry.computeBoundingBox();
    //
    //
    // scene.add(new THREE.Points(geometry, pointmaterial));
    // scene.add(new THREE.Line(geometry, linematerial));

    scene.add(hexahedra_collection);
}

import * as THREE from 'three';

import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DecalGeometry } from 'three/addons/geometries/DecalGeometry.js';

const container = document.getElementById( 'container' );

let renderer, scene, camera, stats;
let mesh;
let raycaster;
let line;
let decalTexture;

const intersection = {
    intersects: false,
    point: new THREE.Vector3(),
    normal: new THREE.Vector3()
};
const mouse = new THREE.Vector2();
const intersects = [];

const textureLoader = new THREE.TextureLoader();

const decals = [];
let mouseHelper;
const position = new THREE.Vector3();
const orientation = new THREE.Euler();
const size = new THREE.Vector3( 30, 30, 30 );

const params = {
    url: "https://avatars.githubusercontent.com/u/13419363?s=280&v=4",  //"put sprite url here",
    rotate: 0.0,
    scale: 0.0,
    undo: function () {

        removeLastDecal();

    },
    clear: function () {

        removeDecals();

    }
};

init();
animate();

function init() {

    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    container.appendChild( renderer.domElement );

    stats = new Stats();
    container.appendChild( stats.dom );

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 1000 );
    camera.position.z = 120;

    const controls = new OrbitControls( camera, renderer.domElement );
    controls.minDistance = 50;
    controls.maxDistance = 200;

    scene.add( new THREE.AmbientLight( 0xffffff ) );

    const geometry = new THREE.BufferGeometry();
    geometry.setFromPoints( [ new THREE.Vector3(), new THREE.Vector3() ] );

    line = new THREE.Line( geometry, new THREE.LineBasicMaterial() );
    scene.add( line );

    loadArc();

    raycaster = new THREE.Raycaster();

    mouseHelper = new THREE.Mesh( new THREE.BoxGeometry( 1, 1, 10 ), new THREE.MeshNormalMaterial() );
    mouseHelper.visible = false;
    scene.add( mouseHelper );

    window.addEventListener( 'resize', onWindowResize );

    let moved = false;

    controls.addEventListener( 'change', function () {

        moved = true;

    } );

    window.addEventListener( 'pointerdown', function () {

        moved = false;

    } );

    window.addEventListener( 'pointerup', function ( event ) {

        if ( moved === false ) {

            checkIntersection( event.clientX, event.clientY );

            if ( intersection.intersects ) shoot();

        }

    } );

    window.addEventListener( 'pointermove', onPointerMove );

    function onPointerMove( event ) {

        if ( event.isPrimary ) {

            checkIntersection( event.clientX, event.clientY );

        }

    }

    function checkIntersection( x, y ) {

        if ( mesh === undefined ) return;

        mouse.x = ( x / window.innerWidth ) * 2 - 1;
        mouse.y = - ( y / window.innerHeight ) * 2 + 1;

        raycaster.setFromCamera( mouse, camera );
        raycaster.intersectObject( mesh, false, intersects );

        if ( intersects.length > 0 ) {

            const p = intersects[ 0 ].point;
            mouseHelper.position.copy( p );
            intersection.point.copy( p );

            const n = intersects[ 0 ].face.normal.clone();
            n.transformDirection( mesh.matrixWorld );
            n.multiplyScalar( 10 );
            n.add( intersects[ 0 ].point );

            intersection.normal.copy( intersects[ 0 ].face.normal );
            mouseHelper.lookAt( n );

            const positions = line.geometry.attributes.position;
            positions.setXYZ( 0, p.x, p.y, p.z );
            positions.setXYZ( 1, n.x, n.y, n.z );
            positions.needsUpdate = true;

            intersection.intersects = true;

            intersects.length = 0;

        } else {

            intersection.intersects = false;

        }

    }

    const gui = new GUI();

    gui.add( params, 'url' );
    gui.add( params, 'rotate', -180, 180);
    gui.add( params, 'scale', -1, 2);
    gui.add( params, 'undo' );
    gui.add( params, 'clear' );
    gui.open();

}

function loadArc() {

    const loader = new GLTFLoader();

    loader.load( 'arc_model/arc_small_lower_quadri_sep.gltf', function ( gltf ) {
        mesh = gltf.scene.children[ 0 ];
        scene.add( mesh );
        mesh.scale.set( 10, 10, 10 );
    } );

}

function shoot() {
    decalTexture = textureLoader.load( params['url'] );
    decalTexture.colorSpace = THREE.SRGBColorSpace;
    position.copy( intersection.point );
    orientation.copy( mouseHelper.rotation );
    orientation.z = orientation.z + params.rotate / 180.0 * Math.PI;
    const scale = 30 * 10 ** params.scale
    const m = new THREE.Mesh(
        new DecalGeometry( mesh, position, orientation, new THREE.Vector3( scale, scale, scale )),
        new THREE.MeshPhongMaterial( {
            map: decalTexture,
            normalScale: new THREE.Vector2( 1, 1 ),
            shininess: 0,
            transparent: true,
            opacity: 0.5,
            depthTest: true,
            depthWrite: false,
            polygonOffset: true,
            polygonOffsetFactor: - 4,
            wireframe: false
        } )
    );

    decals.push( m );
    scene.add( m );

}

function removeLastDecal() {

    scene.remove( decals.pop() )

}

function removeDecals() {

    decals.forEach( function ( d ) {

        scene.remove( d );

    } );

    decals.length = 0;

}

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}

function animate() {

    requestAnimationFrame( animate );

    renderer.render( scene, camera );

    stats.update();

}
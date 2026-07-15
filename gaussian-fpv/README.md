# Kitchen 001 Gaussian FPV

Static GitHub Pages viewer for the RoboGSmart native-v2 kitchen capture. The
scene uses the MIT-licensed [Spark](https://github.com/sparkjsdev/spark) WebGL
Gaussian splat renderer and a version 3 SPZ asset for browser compatibility.

## Controls

- Desktop: click **Enter FPV**, look with the mouse, move with WASD or arrow
  keys, fly vertically with E/Q or Space/C, and hold Shift to boost.
- Touch: drag the left stick to move, drag the scene to look, and use the right
  arrow controls to change altitude.

## Asset

`kitchen-7000.spz` contains 277K Gaussians and three spherical-harmonic bands.
It was converted from the 7,000-step training PLY with the open-source
PlayCanvas `splat-transform` tool:

```sh
splat-transform --spz-version 3 input.ply kitchen-7000.spz
```

SPZ v3 is used because Spark 2.1 supports SPZ versions 1-3. The source PLY was
66 MB; the published SPZ is approximately 4.5 MB.

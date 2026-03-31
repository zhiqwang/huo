# Huo (wip)

## JavaScript Implementation (jax-js)

The `js/` directory contains a JavaScript port of the CT [Algebraic Reconstruction Technique](https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique) (ART), powered by [jax-js](https://github.com/ekzhang/jax-js) for numerical computing in the browser.  Terminology follows the [torch-radon](https://torch-radon.readthedocs.io/en/latest/) convention.

### Quick Start

```bash
cd js
npm install
npx serve .
```

Then open `http://localhost:3000/demo/` in a modern browser (Chrome/Edge with WebGPU recommended).

The demo visualises both the **forward projection** (Radon transform, building the sinogram angle by angle) and the **ART reconstruction** (back-projection, iteratively recovering the image).  A *Frame delay* slider lets you control the speed of each iteration so that both processes can be observed frame by frame.

### Structure

- `js/src/art.js` — Core CT algorithm
  - `forward()` — Radon transform for a single projection angle (image → sinogram column)
  - `backprojection()` — Back-projection for a single angle (sinogram column → image)
  - `scan()` — Full forward projection over all angles (image → complete sinogram)
  - `art()` — ART iterative reconstruction (sinogram → image)
- `js/demo/index.html` — Interactive visualisation page with frame-delay control
- `js/demo/main.js` — Demo setup: phantom generation, frame-by-frame forward projection & reconstruction

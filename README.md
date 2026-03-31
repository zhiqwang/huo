# Huo (wip)

## JavaScript Implementation (jax-js)

The `js/` directory contains a JavaScript port of the CT [Algebraic Reconstruction Technique](https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique) (ART), powered by [jax-js](https://github.com/ekzhang/jax-js) for numerical computing in the browser.

### Quick Start

```bash
cd js
npm install
npx serve .
```

Then open `http://localhost:3000/demo/` in a modern browser (Chrome/Edge with WebGPU recommended).

### Structure

- `js/src/art.js` — Core ART algorithm (forward/backward projection, scan, reconstruction)
- `js/demo/index.html` — Interactive visualization page
- `js/demo/main.js` — Demo setup with phantom generation and live reconstruction

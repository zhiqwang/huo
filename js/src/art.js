// Copyright (c) 2022, Zhiqiang Wang. All rights reserved.
//
// Algebraic Reconstruction Technique (ART) for CT image reconstruction.
// Ported from PyTorch (huo/art.py) to jax-js.
//
// References:
//   - https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique
//   - https://github.com/ekzhang/jax-js

import { numpy } from "@jax-js/jax";

const np = numpy;

/**
 * Bilinear interpolation on a 2D image.
 * Equivalent to PyTorch's F.grid_sample with align_corners=True.
 *
 * @param {Float32Array} imgData - Flattened image data of size H*W.
 * @param {number} H - Image height.
 * @param {number} W - Image width.
 * @param {Float32Array} gridX - X coordinates in [-1, 1] (width axis).
 * @param {Float32Array} gridY - Y coordinates in [-1, 1] (height axis).
 * @returns {Float32Array} Interpolated values with the same length as gridX.
 */
function bilinearSample(imgData, H, W, gridX, gridY) {
  const N = gridX.length;
  const result = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    // Convert normalized [-1, 1] to pixel coordinates
    const px = (gridX[i] + 1) * 0.5 * (W - 1);
    const py = (gridY[i] + 1) * 0.5 * (H - 1);

    const x0 = Math.floor(px);
    const y0 = Math.floor(py);

    const wx = px - x0;
    const wy = py - y0;

    // Clamp indices to valid range
    const cx0 = Math.max(0, Math.min(W - 1, x0));
    const cy0 = Math.max(0, Math.min(H - 1, y0));
    const cx1 = Math.max(0, Math.min(W - 1, x0 + 1));
    const cy1 = Math.max(0, Math.min(H - 1, y0 + 1));

    result[i] =
      imgData[cy0 * W + cx0] * (1 - wy) * (1 - wx) +
      imgData[cy0 * W + cx1] * (1 - wy) * wx +
      imgData[cy1 * W + cx0] * wy * (1 - wx) +
      imgData[cy1 * W + cx1] * wy * wx;
  }

  return result;
}

/**
 * 1D linear interpolation for backward projection.
 * Maps normalized positions in [-1, 1] to data indices [0, N-1].
 *
 * @param {Float32Array} data - 1D data array to interpolate.
 * @param {number} N - Length of data.
 * @param {Float32Array} positions - Normalized positions in [-1, 1].
 * @param {number} count - Number of positions.
 * @returns {Float32Array} Interpolated values.
 */
function linearInterp(data, N, positions, count) {
  const result = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const pos = (positions[i] + 1) * 0.5 * (N - 1);
    const i0 = Math.floor(pos);
    const w = pos - i0;

    const ci0 = Math.max(0, Math.min(N - 1, i0));
    const ci1 = Math.max(0, Math.min(N - 1, i0 + 1));

    result[i] = data[ci0] * (1 - w) + data[ci1] * w;
  }

  return result;
}

/**
 * Fisher-Yates shuffle to generate a random permutation of [0, n).
 *
 * @param {number} n - Length of the permutation.
 * @returns {number[]} Shuffled indices.
 */
function randperm(n) {
  const arr = globalThis.Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Forward propagation per angle.
 *
 * Rotates the gantry coordinates by the view angle, then samples the image
 * along the rotated ray paths using bilinear interpolation. The line integral
 * along each ray is computed by summing across the lateral sampling direction.
 *
 * Equivalent to the PyTorch implementation using F.grid_sample.
 *
 * @param {np.Array} img - 2D image [imgPixels, imgPixels].
 * @param {np.Array} gantryCoordX - Gantry X coordinates [latSteps, detrNum].
 * @param {np.Array} gantryCoordY - Gantry Y coordinates [latSteps, detrNum].
 * @param {number} view - View angle in degrees.
 * @param {Object} param - CT scan parameters.
 * @returns {Promise<np.Array>} Sinogram line for this angle [detrNum].
 */
export async function forwardProjection(img, gantryCoordX, gantryCoordY, view, param) {
  const viewRad = (view * Math.PI) / 180;
  const cosV = Math.cos(viewRad);
  const sinV = Math.sin(viewRad);

  // Read raw data from jax-js arrays (.ref keeps the originals alive)
  const [imgData, gxData, gyData] = await Promise.all([
    img.ref.data(),
    gantryCoordX.ref.data(),
    gantryCoordY.ref.data(),
  ]);

  // Rotate gantry coordinates counter-clockwise
  const N = gxData.length;
  const rotX = new Float32Array(N);
  const rotY = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    rotX[i] = gxData[i] * cosV - gyData[i] * sinV;
    rotY[i] = gxData[i] * sinV + gyData[i] * cosV;
  }

  // Sample image at rotated coordinates
  const interp = bilinearSample(imgData, param.imgPixels, param.imgPixels, rotX, rotY);

  // Sum along lateral direction to compute line integrals
  const latSteps = param.latSampling * param.imgPixels + 1;
  const latStep = param.imgLen / param.imgPixels / param.latSampling;
  const sino = new Float32Array(param.detrNum);

  for (let j = 0; j < param.detrNum; j++) {
    let sum = 0;
    for (let k = 0; k < latSteps; k++) {
      sum += interp[k * param.detrNum + j];
    }
    sino[j] = sum * latStep;
  }

  return np.array(sino);
}

/**
 * Backward propagation per angle.
 *
 * For each pixel in the reconstruction image, computes which detector element
 * it maps to under the given view angle (using fan-beam geometry), then
 * interpolates the sinogram value at that detector position.
 *
 * Equivalent to the PyTorch implementation using F.affine_grid + F.grid_sample.
 *
 * @param {Float32Array} sinogramData - Sinogram data for one angle [detrNum].
 * @param {number} imgEnd - Image coordinate boundary.
 * @param {number} detrEnd - Detector coordinate boundary.
 * @param {number} view - View angle in degrees.
 * @param {Object} param - CT scan parameters.
 * @returns {Promise<np.Array>} Back-projected image [imgPixels, imgPixels].
 */
export async function backwardProjection(sinogramData, imgEnd, detrEnd, view, param) {
  const viewRad = (view * Math.PI) / 180;
  const cosV = Math.cos(viewRad);
  const sinV = Math.sin(viewRad);
  const P = param.imgPixels;

  // For each image pixel, compute its detector coordinate after rotation.
  // This replaces PyTorch's F.affine_grid + F.grid_sample pipeline.
  //
  // Rotation matrix (same as F.affine_grid theta):
  //   rotX = cos*x + sin*y
  //   rotY = -sin*x + cos*y
  //
  // Fan-beam detector mapping:
  //   detCoord = sdd * rotX / (rotY + sod/imgEnd) / detrEnd
  const detPositions = new Float32Array(P * P);

  for (let row = 0; row < P; row++) {
    for (let col = 0; col < P; col++) {
      // Normalized pixel coordinates in [-1, 1]
      const x = -1 + (2 * col) / (P - 1);
      const y = -1 + (2 * row) / (P - 1);

      // Apply rotation
      const rotX = cosV * x + sinV * y;
      const rotY = -sinV * x + cosV * y;

      // Compute detector coordinate (normalized by detrEnd to [-1, 1])
      const adjustedY = rotY + param.sod / imgEnd;
      detPositions[row * P + col] = (param.sdd * rotX) / adjustedY / detrEnd;
    }
  }

  // Interpolate sinogram at computed detector positions
  const imgData = linearInterp(sinogramData, param.detrNum, detPositions, P * P);

  return np.array(imgData).reshape([P, P]);
}

/**
 * CT scanning — forward projection for all view angles.
 *
 * Generates a complete sinogram by running forward projection at each angle.
 *
 * @param {np.Array} img - Input image [imgPixels, imgPixels].
 * @param {np.Array} gantryCoordX - Gantry X coordinates [latSteps, detrNum].
 * @param {np.Array} gantryCoordY - Gantry Y coordinates [latSteps, detrNum].
 * @param {number[]} gantryView - Array of view angles in degrees.
 * @param {Object} param - CT scan parameters.
 * @returns {Promise<np.Array>} Sinogram [detrNum, numViews].
 */
export async function scan(img, gantryCoordX, gantryCoordY, gantryView, param) {
  const numViews = gantryView.length;
  // Store sinogram in [detrNum, numViews] layout (column per view)
  const sinoData = new Float32Array(param.detrNum * numViews);

  for (let i = 0; i < numViews; i++) {
    const sino = await forwardProjection(img, gantryCoordX, gantryCoordY, gantryView[i], param);
    // data() returns the typed array and disposes the jax-js array
    const data = await sino.data();
    for (let j = 0; j < param.detrNum; j++) {
      sinoData[j * numViews + i] = data[j];
    }
  }

  return np.array(sinoData).reshape([param.detrNum, numViews]);
}

/**
 * Algebraic Reconstruction Technique (ART).
 *
 * Iteratively reconstructs a CT image from sinogram data by processing one
 * view angle at a time in random order. For each view:
 *   1. Forward project the current estimate to predict the sinogram.
 *   2. Compute the residual (measured - predicted).
 *   3. Back-project the residual to update the image estimate.
 *   4. Clamp negative values to zero.
 *
 * @param {np.Array} sinogram - Sinogram data [detrNum, numViews].
 * @param {number} imgEnd - Image coordinate boundary.
 * @param {number} detrEnd - Detector coordinate boundary.
 * @param {np.Array} gantryCoordX - Gantry X coordinates [latSteps, detrNum].
 * @param {np.Array} gantryCoordY - Gantry Y coordinates [latSteps, detrNum].
 * @param {number[]} gantryView - Array of view angles in degrees.
 * @param {Object} param - CT scan parameters.
 * @param {Function} [onProgress] - Async callback for visualization updates.
 *   Called with (imgData: Float32Array, imgSize: number, viewIdx: number).
 * @returns {Promise<np.Array>} Reconstructed image [imgPixels, imgPixels].
 */
export async function art(
  sinogram,
  imgEnd,
  detrEnd,
  gantryCoordX,
  gantryCoordY,
  gantryView,
  param,
  onProgress,
) {
  const numViews = gantryView.length;
  const P = param.imgPixels;

  // Initialize reconstruction to zeros
  let imgData = new Float32Array(P * P);

  // Read sinogram data (.ref keeps the original alive)
  const fullSinoData = await sinogram.ref.data();

  // Process views in random order (Kaczmarz method)
  const indices = randperm(numViews);

  for (const idx of indices) {
    const view = gantryView[idx];

    // Create jax-js array for current image estimate
    const img = np.array(imgData).reshape([P, P]);

    // Forward project current estimate
    const fp = await forwardProjection(img, gantryCoordX, gantryCoordY, view, param);
    const fpData = await fp.data();

    // Extract measured sinogram column for this view
    const sinoCol = new Float32Array(param.detrNum);
    for (let j = 0; j < param.detrNum; j++) {
      sinoCol[j] = fullSinoData[j * numViews + idx];
    }

    // Compute residual: (measured - predicted) / imgLen
    const resData = new Float32Array(param.detrNum);
    for (let j = 0; j < param.detrNum; j++) {
      resData[j] = (sinoCol[j] - fpData[j]) / param.imgLen;
    }

    // Back-project residual into image space
    const bp = await backwardProjection(resData, imgEnd, detrEnd, view, param);
    const bpData = await bp.data();

    // Update image estimate: img += backprojection, then clamp to [0, +inf)
    for (let i = 0; i < P * P; i++) {
      imgData[i] = Math.max(0, imgData[i] + bpData[i]);
    }

    // Dispose the temporary image array
    img.dispose();

    // Notify progress for visualization
    if (onProgress) {
      await onProgress(imgData, P, idx);
    }
  }

  return np.array(imgData).reshape([P, P]);
}

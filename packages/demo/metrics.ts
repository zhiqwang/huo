export interface ImageMetrics {
  relativeL2: number;
  psnr: number;
  ssim: number;
  flatVariation: number | undefined;
}

// Wang et al.'s 11 × 11 Gaussian window, with sigma 1.5.
const windowRadius = 5;
const weights = Float64Array.from({ length: 2 * windowRadius + 1 }, (_, i) =>
  Math.exp(-0.5 * ((i - windowRadius) / 1.5) ** 2),
);
const weightSum = weights.reduce((sum, weight) => sum + weight, 0);
for (let i = 0; i < weights.length; i++) weights[i] /= weightSum;

function gaussianMean(image: Float32Array | Float64Array, size: number): Float64Array {
  const width = size - 2 * windowRadius;
  const rows = new Float64Array(size * width);
  const means = new Float64Array(width * width);
  // Two separable passes over windows contained entirely in the image.
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < width; col++) {
      let value = 0;
      for (let i = 0; i < weights.length; i++) value += weights[i] * image[row * size + col + i];
      rows[row * width + col] = value;
    }
  }
  for (let row = 0; row < width; row++) {
    for (let col = 0; col < width; col++) {
      let value = 0;
      for (let i = 0; i < weights.length; i++) value += weights[i] * rows[(row + i) * width + col];
      means[row * width + col] = value;
    }
  }
  return means;
}

function flatPairs(reference: Float64Array, size: number): [number, number][] {
  const edge = new Uint8Array(reference.length);
  const nearEdge = new Uint8Array(reference.length);
  const tolerance = 1e-6;
  for (let i = 0; i < reference.length; i++) {
    if (i % size < size - 1 && Math.abs(reference[i] - reference[i + 1]) > tolerance) {
      edge[i] = edge[i + 1] = 1;
    }
    if (i + size < reference.length && Math.abs(reference[i] - reference[i + size]) > tolerance) {
      edge[i] = edge[i + size] = 1;
    }
  }
  for (let i = 0; i < edge.length; i++) {
    if (!edge[i]) continue;
    const row = Math.floor(i / size);
    const col = i % size;
    for (let y = Math.max(0, row - 1); y <= Math.min(size - 1, row + 1); y++) {
      const start = y * size + Math.max(0, col - 1);
      nearEdge.fill(1, start, y * size + Math.min(size, col + 2));
    }
  }
  const interior = Uint8Array.from(reference, (pixel, i) => Number(pixel > tolerance && !nearEdge[i]));
  const pairs: [number, number][] = [];
  for (let i = 0; i < interior.length; i++) {
    if (!interior[i]) continue;
    if (i % size < size - 1 && interior[i + 1]) pairs.push([i, i + 1]);
    if (interior[i + size]) pairs.push([i, i + size]);
  }
  return pairs;
}

/** Measure raw snapshots against a fixed phantom and its intensity range. */
export function createImageMetrics(phantom: Float32Array, size: number): (image: Float32Array) => ImageMetrics {
  if (!Number.isInteger(size) || size < weights.length || phantom.length !== size * size)
    throw new RangeError("the phantom must be a square image at least 11 pixels wide");
  const reference = Float64Array.from(phantom);
  let minimum = Infinity;
  let peak = -Infinity;
  let energy = 0;
  for (const pixel of reference) {
    minimum = Math.min(minimum, pixel);
    peak = Math.max(peak, pixel);
    energy += pixel ** 2;
  }
  const range = peak - minimum;
  if (!Number.isFinite(range) || range <= 0)
    throw new RangeError("the phantom must have a finite, positive intensity range");

  const referenceMean = gaussianMean(reference, size);
  const referenceVariance = gaussianMean(reference.map((pixel) => pixel ** 2), size)
    .map((moment, i) => moment - referenceMean[i] ** 2);
  const c1 = (0.01 * range) ** 2;
  const c2 = (0.03 * range) ** 2;
  const pairs = flatPairs(reference, size);

  return (image) => {
    if (image.length !== reference.length) throw new RangeError("image must match the phantom shape");
    const squaredError = image.reduce((sum, pixel, i) => sum + (pixel - reference[i]) ** 2, 0);
    const mean = gaussianMean(image, size);
    const squaredMean = gaussianMean(Float64Array.from(image, (pixel) => pixel ** 2), size);
    const productMean = gaussianMean(Float64Array.from(image, (pixel, i) => pixel * reference[i]), size);
    let similarity = 0;
    for (let i = 0; i < mean.length; i++) {
      const variance = squaredMean[i] - mean[i] ** 2;
      const covariance = productMean[i] - mean[i] * referenceMean[i];
      similarity += (2 * mean[i] * referenceMean[i] + c1) * (2 * covariance + c2) /
        ((mean[i] ** 2 + referenceMean[i] ** 2 + c1) * (variance + referenceVariance[i] + c2));
    }
    const variation = pairs.reduce((sum, [i, j]) => sum + (image[i] - image[j]) ** 2, 0);
    return {
      relativeL2: 100 * Math.sqrt(squaredError / energy),
      psnr: 20 * Math.log10(range / Math.sqrt(squaredError / reference.length)),
      ssim: similarity / mean.length,
      flatVariation: pairs.length ? 100 * Math.sqrt(variation / pairs.length) / peak : undefined,
    };
  };
}

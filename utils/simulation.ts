import { TrainingSample } from '../types';

// Simplified Projection for Visualization (2D plot of features)
// We map high-dimensional EEM data to 2D using Center of Mass (Peak position in this implementation).
// This preserves the "topology" relation you mentioned.
export const calculateProjection = (samples: TrainingSample[]) => {
  if (samples.length === 0) return [];

  const xValues = samples.map(s => s.features.peakEm);
  const yValues = samples.map(s => s.features.peakEx); // Excitation usually Y axis in EEMs

  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);

  return samples.map(s => ({
    ...s,
    // Normalize to 0-100 for charting, with some jitter to see overlapping points
    x: ((s.features.peakEm - minX) / (maxX - minX || 1)) * 100, 
    y: ((s.features.peakEx - minY) / (maxY - minY || 1)) * 100,
  }));
};
import { EEMPoint, EEMFeatures, TrainingSample, PredictionResult, TrainedModel } from '../types';

// ==========================================
// 0. 数据预处理
// ==========================================

const removeRayleighScatter = (data: EEMPoint[]): EEMPoint[] => {
  const bandwidth1st = 20; 
  const bandwidth2nd = 20; 

  return data.map(p => {
    let intensity = p.intensity;
    if (Math.abs(p.ex - p.em) < bandwidth1st) intensity = 0;
    if (Math.abs(p.em - 2 * p.ex) < bandwidth2nd) intensity = 0;
    return { ...p, intensity };
  });
};

// ==========================================
// 1. 特征提取核心算法 (Advanced)
// ==========================================

const MIN_EX = 200, MAX_EX = 500;
const MIN_EM = 250, MAX_EM = 600;

// 计算网格特征
const calculateGridFeatures = (data: EEMPoint[], gridRows: number, gridCols: number): number[] => {
  const exStep = (MAX_EX - MIN_EX) / gridRows;
  const emStep = (MAX_EM - MIN_EM) / gridCols;
  
  const grid = new Array(gridRows * gridCols).fill(0);
  
  for (const p of data) {
    if (p.ex < MIN_EX || p.ex > MAX_EX || p.em < MIN_EM || p.em > MAX_EM) continue;
    
    const row = Math.min(Math.floor((p.ex - MIN_EX) / exStep), gridRows - 1);
    const col = Math.min(Math.floor((p.em - MIN_EM) / emStep), gridCols - 1);
    const idx = row * gridCols + col;
    
    // Max pooling for shape preservation
    grid[idx] = Math.max(grid[idx], p.intensity);
  }
  
  return grid;
};

// 计算行列轮廓 (Row/Col Profiles: Mean & Max)
const calculateProfiles = (data: EEMPoint[]): { 
  rowMean: number[], colMean: number[], rowMax: number[], colMax: number[] 
} => {
  const bins = 50; // High resolution profiles
  
  const rowSum = new Array(bins).fill(0);
  const rowCount = new Array(bins).fill(0);
  const rowMax = new Array(bins).fill(-Infinity);

  const colSum = new Array(bins).fill(0);
  const colCount = new Array(bins).fill(0);
  const colMax = new Array(bins).fill(-Infinity);

  const exStep = (MAX_EX - MIN_EX) / bins;
  const emStep = (MAX_EM - MIN_EM) / bins;

  for (const p of data) {
    const val = Math.max(0, p.intensity);

    // Row (Ex)
    if (p.ex >= MIN_EX && p.ex <= MAX_EX) {
      const i = Math.min(Math.floor((p.ex - MIN_EX) / exStep), bins - 1);
      rowSum[i] += val;
      rowCount[i]++;
      if (val > rowMax[i]) rowMax[i] = val;
    }

    // Col (Em)
    if (p.em >= MIN_EM && p.em <= MAX_EM) {
      const j = Math.min(Math.floor((p.em - MIN_EM) / emStep), bins - 1);
      colSum[j] += val;
      colCount[j]++;
      if (val > colMax[j]) colMax[j] = val;
    }
  }

  // Clean up
  const rowMean = rowSum.map((s, i) => rowCount[i] > 0 ? s / rowCount[i] : 0);
  const colMean = colSum.map((s, i) => colCount[i] > 0 ? s / colCount[i] : 0);
  const safeRowMax = rowMax.map(v => v === -Infinity ? 0 : v);
  const safeColMax = colMax.map(v => v === -Infinity ? 0 : v);

  return { rowMean, colMean, rowMax: safeRowMax, colMax: safeColMax };
};

const calculateAdvancedStats = (data: EEMPoint[]) => {
  const intensities = data.map(d => Math.max(0, d.intensity));
  const n = intensities.length;
  if (n === 0) return { mean: 0, stdDev: 0, skewness: 0, kurtosis: 0, max: 0, entropy: 0, sparsity: 1 };

  const max = Math.max(...intensities);
  const sum = intensities.reduce((a, b) => a + b, 0);
  const mean = sum / n;

  // StdDev, Skew, Kurtosis
  let sumSqDiff = 0;
  let sumCuDiff = 0;
  let sumQuDiff = 0;
  let zeroCount = 0;

  // For Entropy
  const hist = new Map<number, number>();
  
  for (const val of intensities) {
    const diff = val - mean;
    sumSqDiff += Math.pow(diff, 2);
    sumCuDiff += Math.pow(diff, 3);
    sumQuDiff += Math.pow(diff, 4);
    
    if (val < 0.1) zeroCount++; // Threshold for sparsity

    // Simple binning for entropy
    const bin = Math.floor(val);
    hist.set(bin, (hist.get(bin) || 0) + 1);
  }

  const variance = sumSqDiff / n;
  const stdDev = Math.sqrt(variance);
  
  const skewness = stdDev > 0 ? (sumCuDiff / n) / Math.pow(stdDev, 3) : 0;
  const kurtosis = stdDev > 0 ? (sumQuDiff / n) / Math.pow(stdDev, 4) : 0;
  const sparsity = zeroCount / n;

  // Calculate Entropy: -Sum(p * log(p))
  let entropy = 0;
  for (const count of hist.values()) {
    const p = count / n;
    if (p > 0) entropy -= p * Math.log2(p);
  }

  return { mean, stdDev, skewness, kurtosis, max, entropy, sparsity };
};

const calculateCenterOfMass = (data: EEMPoint[]) => {
  let maxVal = -Infinity;
  let peakEx = 0;
  let peakEm = 0;

  for (const p of data) {
    const val = Math.max(0, p.intensity);
    if (val > maxVal) {
      maxVal = val;
      peakEx = p.ex;
      peakEm = p.em;
    }
  }
  
  if (maxVal === -Infinity) return { peakEx: 0, peakEm: 0 };
  return { peakEx, peakEm };
};

export const extractFeatures = (rawData: EEMPoint[]): EEMFeatures => {
  const cleanData = removeRayleighScatter(rawData);

  // 1. 全局统计
  const stats = calculateAdvancedStats(cleanData);
  
  // 2. 峰值定位
  const com = calculateCenterOfMass(cleanData);
  
  // 3. 区域积分 (3x3)
  const riaGrid = calculateGridFeatures(cleanData, 3, 3);
  
  // 4. 行列轮廓 (Profiles)
  const profiles = calculateProfiles(cleanData);
  
  // 5. 形状网格 (用于 PCA)
  // OPTIMIZATION: Reduced from 50x50 (2500 dims) to 15x15 (225 dims) 
  // to fix browser hanging during PCA covariance matrix calculation.
  const shapeGrid = calculateGridFeatures(cleanData, 15, 15);

  return {
    maxIntensity: stats.max,
    mean: stats.mean,
    stdDev: stats.stdDev,
    skewness: stats.skewness,
    kurtosis: stats.kurtosis,
    entropy: stats.entropy,
    sparsity: stats.sparsity,
    
    peakEx: com.peakEx,
    peakEm: com.peakEm,
    
    rowMeanProfile: profiles.rowMean,
    colMeanProfile: profiles.colMean,
    rowMaxProfile: profiles.rowMax,
    colMaxProfile: profiles.colMax,
    
    riaGrid: riaGrid,
    shapeGrid: shapeGrid,
    combinedVector: [] 
  };
};

// ==========================================
// 2. PCA 引擎 (优化版)
// ==========================================

const dotProduct = (a: number[], b: number[]) => a.reduce((sum, v, i) => sum + v * b[i], 0);

const trainPCA = (vectors: number[][], nComponents: number): { matrix: number[][], means: number[], stds: number[] } => {
  const n = vectors.length;
  if (n === 0) return { matrix: [], means: [], stds: [] };
  
  const dim = vectors[0].length;
  const actualComponents = Math.min(nComponents, n - 1, dim);

  // 1. Z-Score 标准化
  const means = new Array(dim).fill(0);
  const stds = new Array(dim).fill(0);

  for (let j = 0; j < dim; j++) {
    means[j] = vectors.reduce((sum, v) => sum + v[j], 0) / n;
  }
  for (let j = 0; j < dim; j++) {
    const variance = vectors.reduce((sum, v) => sum + Math.pow(v[j] - means[j], 2), 0) / n;
    stds[j] = Math.sqrt(variance) || 1; 
  }

  const standardized = vectors.map(v => v.map((val, j) => (val - means[j]) / stds[j]));
  
  // 2. 计算协方差矩阵 (Covariance Matrix)
  const covMatrix = new Array(dim).fill(0).map(() => new Array(dim).fill(0));
  for (let i = 0; i < dim; i++) {
    for (let j = i; j < dim; j++) { // 对称矩阵，只计算一半
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += standardized[k][i] * standardized[k][j];
      }
      const val = sum / (n - 1 || 1);
      covMatrix[i][j] = val;
      covMatrix[j][i] = val;
    }
  }

  // 3. 幂迭代法求特征向量 (带 Deflation)
  const eigenvectors: number[][] = [];
  const currentCov = covMatrix.map(row => [...row]); 

  for (let c = 0; c < actualComponents; c++) {
    // 随机初始化向量
    let vec = new Array(dim).fill(0).map(() => Math.random() - 0.5);
    let mag = Math.sqrt(dotProduct(vec, vec));
    vec = vec.map(v => v / (mag || 1e-6));

    // 迭代寻找主成分
    for (let iter = 0; iter < 100; iter++) { 
       const nextVec = new Array(dim).fill(0);
       for(let i=0; i<dim; i++) {
         for(let j=0; j<dim; j++) {
           nextVec[i] += currentCov[i][j] * vec[j];
         }
       }
       mag = Math.sqrt(dotProduct(nextVec, nextVec));
       if (mag < 1e-10) break; // 避免除零
       vec = nextVec.map(v => v / mag);
    }
    eigenvectors.push(vec);

    // Deflation (矩阵收缩): C_new = C_old - lambda * v * v^T
    // 计算特征值 lambda = v^T * C * v
    const Cv = new Array(dim).fill(0);
    for(let i=0; i<dim; i++) {
        for(let j=0; j<dim; j++) {
           Cv[i] += currentCov[i][j] * vec[j];
        }
    }
    const lambda = dotProduct(vec, Cv);

    // 更新协方差矩阵，移除当前维度的影响
    for(let i=0; i<dim; i++) {
        for(let j=0; j<dim; j++) {
            currentCov[i][j] -= lambda * vec[i] * vec[j];
        }
    }
  }

  // 转置特征向量矩阵，以便 applyPCA 使用
  const projectionMatrix = new Array(dim).fill(0).map((_, i) => eigenvectors.map(row => row[i] || 0));
  return { matrix: projectionMatrix, means, stds };
};

const applyPCA = (vector: number[], model: { matrix: number[][], means: number[], stds: number[] }): number[] => {
  const standardized = vector.map((v, i) => (v - model.means[i]) / model.stds[i]);
  const numComponents = model.matrix.length > 0 ? model.matrix[0].length : 0;
  const result = new Array(numComponents).fill(0);
  
  for (let j = 0; j < numComponents; j++) {
    for (let i = 0; i < vector.length; i++) {
      result[j] += standardized[i] * model.matrix[i][j];
    }
  }
  return result;
};

// ==========================================
// 3. 模型训练与保存
// ==========================================

const MODEL_KEY = 'fluor_detect_model_v7_optimized'; 

const calculateNormalizationParams = (vectors: number[][]) => {
  const n = vectors.length;
  if (n === 0) return { means: [], stds: [] };
  const dim = vectors[0].length;
  const means = new Array(dim).fill(0);
  const stds = new Array(dim).fill(0);

  for (let j = 0; j < dim; j++) {
    means[j] = vectors.reduce((sum, v) => sum + v[j], 0) / n;
  }
  for (let j = 0; j < dim; j++) {
    const variance = vectors.reduce((sum, v) => sum + Math.pow(v[j] - means[j], 2), 0) / n;
    stds[j] = Math.sqrt(variance) || 1e-6; 
  }
  return { means, stds };
}

// 保存草稿状态（只存文件列表，不存PCA模型参数）
export const saveDraftModel = (samples: TrainingSample[]) => {
  const draftModel: TrainedModel = {
    samples: samples.map(s => ({
      ...s,
      data: [], 
      x: undefined, 
      y: undefined,
      finalVector: []
    })),
    pcaMatrix: [],
    pcaMeans: [],
    pcaStds: [],
    finalMean: [],
    finalStd: [],
    timestamp: Date.now()
  };
  
  try {
    localStorage.setItem(MODEL_KEY, JSON.stringify(draftModel));
  } catch (e) {
    console.error("草稿保存失败", e);
  }
};

export const trainAndSaveModel = (samples: TrainingSample[], modelName: string = "Default Model"): TrainedModel => {
  if (samples.length === 0) throw new Error("无训练数据");

  const mutableSamples = samples.map(s => ({
    ...s,
    features: { ...s.features }, 
  }));

  // PCA 输入: 
  const vectorsForPCA = mutableSamples.map(s => [
      ...s.features.shapeGrid,
      ...s.features.rowMeanProfile,
      ...s.features.colMeanProfile,
      ...s.features.rowMaxProfile,
      ...s.features.colMaxProfile
  ]);
  
  // PCA Components = 20
  const pcaModel = trainPCA(vectorsForPCA, 20); 

  const combinedVectors: number[][] = mutableSamples.map((s, i) => {
    const explicitStats = [
      s.features.maxIntensity, 
      s.features.mean, 
      s.features.stdDev,
      s.features.skewness,
      s.features.kurtosis,
      s.features.entropy, 
      s.features.sparsity 
    ];
    
    const peakFeatures = [s.features.peakEx, s.features.peakEm];
    
    const ria = s.features.riaGrid; 
    const pcaFeatures = applyPCA(vectorsForPCA[i], pcaModel); 
    
    // 使用 PC1 和 PC2 作为可视化坐标
    s.x = pcaFeatures.length > 0 ? pcaFeatures[0] : 0;
    s.y = pcaFeatures.length > 1 ? pcaFeatures[1] : 0;

    return [...explicitStats, ...peakFeatures, ...ria, ...pcaFeatures];
  });

  const { means: finalMean, stds: finalStd } = calculateNormalizationParams(combinedVectors);

  mutableSamples.forEach((s, i) => {
    s.features.combinedVector = combinedVectors[i];
    s.finalVector = combinedVectors[i].map((val, idx) => (val - finalMean[idx]) / finalStd[idx]);
  });

  const model: TrainedModel = {
    name: modelName,
    samples: mutableSamples,
    pcaMatrix: pcaModel.matrix,
    pcaMeans: pcaModel.means,
    pcaStds: pcaModel.stds,
    finalMean,
    finalStd,
    timestamp: Date.now()
  };

  const modelToSave: TrainedModel = {
    ...model,
    samples: model.samples.map(s => ({
      ...s,
      data: [] 
    }))
  };

  try {
    localStorage.setItem(MODEL_KEY, JSON.stringify(modelToSave));
  } catch (e) {
    console.error("模型保存失败", e);
  }

  return model;
};

export const loadModel = (): TrainedModel | null => {
  try {
    const str = localStorage.getItem(MODEL_KEY);
    if (!str) return null;
    return JSON.parse(str) as TrainedModel;
  } catch (e) {
    console.error("加载模型失败", e);
    return null;
  }
};

// ==========================================
// 4. 预测 (Inference)
// ==========================================

const euclideanDist = (v1: number[], v2: number[]) => {
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += Math.pow(v1[i] - v2[i], 2);
  }
  return Math.sqrt(sum);
};

export const predictSample = (
  unknownData: EEMPoint[], 
  model: TrainedModel, 
  fileName: string,
  k: number = 3
): PredictionResult => {
  const features = extractFeatures(unknownData);
  
  const vectorForPCA = [
      ...features.shapeGrid,
      ...features.rowMeanProfile,
      ...features.colMeanProfile,
      ...features.rowMaxProfile,
      ...features.colMaxProfile
  ];

  const pcaModel = { matrix: model.pcaMatrix, means: model.pcaMeans, stds: model.pcaStds };
  const pcaFeatures = applyPCA(vectorForPCA, pcaModel);

  const explicitStats = [
      features.maxIntensity, features.mean, features.stdDev, features.skewness, features.kurtosis,
      features.entropy, features.sparsity
  ];
  const peakFeatures = [features.peakEx, features.peakEm];

  const rawCombined = [...explicitStats, ...peakFeatures, ...features.riaGrid, ...pcaFeatures];
  features.combinedVector = rawCombined;

  // 安全检查
  const safeCombined = rawCombined.length === model.finalMean.length 
      ? rawCombined 
      : rawCombined.slice(0, model.finalMean.length);

  const normalizedVector = safeCombined.map((val, i) => (val - model.finalMean[i]) / model.finalStd[i]);

  const distances = model.samples.map(sample => ({
    sample,
    dist: euclideanDist(normalizedVector, sample.finalVector)
  }));

  distances.sort((a, b) => a.dist - b.dist);
  const neighbors = distances.slice(0, k);

  const votes: Record<string, number> = {};
  neighbors.forEach(n => {
    const weight = 1 / (n.dist + 1e-6); 
    votes[n.sample.label] = (votes[n.sample.label] || 0) + weight;
  });

  let winnerLabel = "Unknown";
  let maxVote = -1;
  let totalScore = 0;
  for (const [label, score] of Object.entries(votes)) {
    totalScore += score;
    if (score > maxVote) {
      maxVote = score;
      winnerLabel = label;
    }
  }

  let ratioSum = 0;
  let weightSum = 0;
  neighbors.forEach(n => {
    const weight = 1 / (n.dist + 1e-6);
    ratioSum += n.sample.ratio * weight;
    weightSum += weight;
  });
  const predictedRatio = weightSum > 0 ? ratioSum / weightSum : 0;

  return {
    fileName,
    predictedLabel: winnerLabel,
    predictedRatio: predictedRatio,
    confidence: totalScore > 0 ? (maxVote / totalScore) * 100 : 0,
    nearestNeighbors: neighbors.map(n => n.sample),
    sampleFeatures: features,
    distances: neighbors.map(n => n.dist)
  };
};

export const parseEEMFile = async (file: File): Promise<{ data: EEMPoint[], label: string, ratio: number }> => {
  const text = await file.text();
  const lines = text.trim().split('\n');
  const data: EEMPoint[] = [];

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 3) {
      const em = parseFloat(parts[0]); 
      const ex = parseFloat(parts[1]); 
      const int = parseFloat(parts[2]);
      if (!isNaN(ex) && !isNaN(em) && !isNaN(int)) {
        data.push({ ex, em, intensity: int });
      }
    }
  }

  const nameParts = file.name.replace('.txt', '').split('-');
  let label = "未知";
  let ratio = 0;
  if (nameParts.length >= 2) {
    label = nameParts[0];
    ratio = parseFloat(nameParts[1]);
  } else {
    label = file.name;
  }
  return { data, label, ratio };
};


export interface EEMPoint {
  ex: number;
  em: number;
  intensity: number;
}

export interface EEMFeatures {
  // --- 1. 扩展全局统计特征 (7维) ---
  maxIntensity: number;
  mean: number;
  stdDev: number;
  skewness: number;
  kurtosis: number;
  entropy: number; // 新增: 信息熵 (复杂度)
  sparsity: number; // 新增: 稀疏度 (零值占比)
  
  // --- 2. 峰值定位 (2维) ---
  peakEx: number; 
  peakEm: number; 

  // --- 3. 行列轮廓特征 (用于 PCA 输入) ---
  // 捕捉 Ex 和 Em 方向的平均趋势和最大趋势
  rowMeanProfile: number[]; // Ex 方向均值
  colMeanProfile: number[]; // Em 方向均值
  rowMaxProfile: number[];  // Ex 方向最大值
  colMaxProfile: number[];  // Em 方向最大值

  // --- 4. 区域积分特征 3x3 (9维) ---
  riaGrid: number[]; 

  // --- 5. 形态特征网格 (High Res 50x50) ---
  shapeGrid: number[];
  
  // --- 6. 最终混合特征向量 (用于距离计算) ---
  combinedVector: number[];
}

// 保存的训练模型结构
export interface TrainedModel {
  name?: string; // 新增: 模型名称
  samples: TrainingSample[]; 
  
  // PCA 参数
  pcaMatrix: number[][]; 
  pcaMeans: number[]; 
  pcaStds: number[]; 
  
  // Z-Score 标准化参数
  finalMean: number[];
  finalStd: number[];
  
  timestamp: number;
}

export interface TrainingSample {
  id: string;
  fileName: string;
  label: string; 
  ratio: number; 
  data: EEMPoint[]; 
  features: EEMFeatures; 
  finalVector: number[];
  // Visualization coordinates (PC1, PC2) - 只在训练后生成
  x?: number;
  y?: number;
}

export interface PredictionResult {
  fileName: string;
  predictedLabel: string;
  predictedRatio: number; 
  confidence: number;
  nearestNeighbors: TrainingSample[];
  sampleFeatures: EEMFeatures;
  distances: number[];
}

export enum AppView {
  TRAINING = 'TRAINING',
  DETECTION = 'DETECTION'
}

export enum DetectionStatus {
  IDLE = '准备预测',
  PREDICTED = '已预测'
}

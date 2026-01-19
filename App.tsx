import React, { useState, useMemo, useRef, useEffect } from 'react';
import { 
  UploadCloud, 
  FileText, 
  BrainCircuit,
  BarChart3,
  Microscope,
  Play,
  RefreshCw,
  FolderUp,
  Activity,
  List,
  AlertCircle,
  XCircle,
  Download,
  Database,
  Eraser,
  Save,
  FileUp,
  FileDown,
  Trash2,
  Clock,
  CheckCircle, 
  AlertTriangle,
  Layout,
  Layers
} from 'lucide-react';
import { AppView, DetectionStatus, TrainingSample, PredictionResult, TrainedModel } from './types';
import { parseEEMFile, extractFeatures, predictSample, trainAndSaveModel, loadModel, saveDraftModel } from './utils/mlEngine';
import { EEMHeatmap } from './components/EEMHeatmap';
import { ScatterChart as ReScatter, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function App() {
  // ================= State Management =================
  const [currentView, setCurrentView] = useState<AppView>(AppView.TRAINING);
  
  // Core Data
  const [trainingData, setTrainingData] = useState<TrainingSample[]>([]);
  const [trainedModel, setTrainedModel] = useState<TrainedModel | null>(null);
  
  // Model Management
  const [modelName, setModelName] = useState("My_EEM_Model");
  
  // Training Mode
  const [trainingStrategy, setTrainingStrategy] = useState<'append' | 'new'>('append');

  // Selection & UI
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Detection Mode Data
  const [detectionStatus, setDetectionStatus] = useState<DetectionStatus>(DetectionStatus.IDLE);
  const [testFiles, setTestFiles] = useState<File[]>([]); 
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [selectedPredictionIdx, setSelectedPredictionIdx] = useState<number | null>(null);

  // Refs
  const trainingInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const importModelInputRef = useRef<HTMLInputElement>(null);
  
  const testInputRef = useRef<HTMLInputElement>(null);
  const testFolderInputRef = useRef<HTMLInputElement>(null);

  // ================= Lifecycle =================
  useEffect(() => {
    try {
        // App 启动时加载持久化的模型
        const saved = loadModel();
        if (saved) {
          restoreModel(saved);
        }
    } catch (e) {
        console.error("Failed to load model on startup:", e);
    }
  }, []);

  const restoreModel = (saved: TrainedModel) => {
      // Safety check to prevent white screen if data is corrupted
      if (!saved || !Array.isArray(saved.samples)) {
          console.warn("Invalid model data found in storage.");
          return;
      }
      
      const isTrained = saved.pcaMatrix && saved.pcaMatrix.length > 0;
      
      // 恢复 ID (防止 ID 丢失)
      const uniqueSamples: TrainingSample[] = [];
      const seenIds = new Set<string>();
      
      saved.samples.forEach(s => {
         let safeId = s.id;
         if (!safeId || seenIds.has(safeId)) {
             safeId = window.crypto.randomUUID ? window.crypto.randomUUID() : `restored_${Date.now()}_${Math.random().toString(36).substr(2,9)}`;
             s.id = safeId;
         }
         seenIds.add(safeId);
         uniqueSamples.push(s);
      });

      // 排序: 自然排序 (1, 2, 10)
      uniqueSamples.sort((a, b) => a.fileName.localeCompare(b.fileName, undefined, { numeric: true, sensitivity: 'base' }));

      setTrainingData(uniqueSamples);
      if (saved.name) setModelName(saved.name);
      
      if (isTrained) {
        setTrainedModel({ ...saved, samples: uniqueSamples });
      } else {
        setTrainedModel(null);
      }
      
      if(uniqueSamples.length > 0) setSelectedSampleId(uniqueSamples[0].id);
  }

  // ================= Handlers =================

  // --- Model Management ---
  const handleExportModel = () => {
    if (!trainedModel) {
      alert("当前没有已训练的模型可导出。");
      return;
    }
    const jsonStr = JSON.stringify(trainedModel);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${trainedModel.name || "model"}_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleImportModel = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setIsProcessing(true);
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target?.result as string);
        if (json.samples && json.pcaMatrix) {
             restoreModel(json as TrainedModel);
             alert(`模型 "${json.name || 'Unknown'}" 导入成功！\n现在您可以直接进入【水样检测】界面使用该模型。`);
        } else {
             alert("文件格式不正确，不是有效的模型文件。");
        }
      } catch (err) {
        console.error(err);
        alert("模型导入失败：文件损坏或格式错误。");
      } finally {
        setIsProcessing(false);
        if (importModelInputRef.current) importModelInputRef.current.value = "";
      }
    };
    reader.readAsText(file);
  };

  const handleClearTrainingData = () => {
      if (trainingData.length === 0) return;
      
      // 直接清空，不再弹窗询问，提升操作流畅度
      setTrainingData([]);
      setSelectedSampleId(null);
      saveDraftModel([]); 

      if (trainingInputRef.current) trainingInputRef.current.value = "";
      if (folderInputRef.current) folderInputRef.current.value = "";
  };
  
  const handleRemoveTrainingSample = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const newData = trainingData.filter(s => s.id !== id);
    setTrainingData(newData);
    saveDraftModel(newData);
    if (selectedSampleId === id) {
        setSelectedSampleId(null);
    }
  };

  // 1. Data Import (Training)
  const handleTrainingUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    setIsProcessing(true);
    
    // Explicitly cast to File[] to avoid 'unknown' type issues with Array.from on FileList
    const files = (Array.from(e.target.files) as File[]).filter(f => f.name.toLowerCase().endsWith('.txt'));
    
    if (files.length === 0) {
      alert("未找到 .txt 文件");
      setIsProcessing(false);
      return;
    }

    try {
        const promises = files.map(file => parseEEMFile(file));
        const parsedResults = await Promise.all(promises);
        
        const newSamples: TrainingSample[] = [];
        let skippedCount = 0;

        const baseData = trainingStrategy === 'new' ? [] : trainingData;
        const existingNames = new Set(baseData.map(s => s.fileName));

        parsedResults.forEach((res, index) => {
            const fileName = files[index].name;
            
            if (existingNames.has(fileName)) {
                skippedCount++;
                return; 
            }

            const { data, label, ratio } = res;
            if (data.length > 0) {
                 const features = extractFeatures(data);
                 const uniqueId = window.crypto.randomUUID 
                    ? window.crypto.randomUUID() 
                    : `sample_${Date.now()}_${index}_${Math.random().toString(36).substr(2,9)}`;
                 
                 newSamples.push({
                    id: uniqueId,
                    fileName,
                    label,
                    ratio,
                    data,
                    features,
                    finalVector: []
                 });
            }
        });

        if (skippedCount > 0 && trainingStrategy === 'append') {
           alert(`${skippedCount} 个文件因重名被跳过。`);
        }

        if (newSamples.length === 0 && skippedCount === files.length) {
            setIsProcessing(false);
            if (trainingInputRef.current) trainingInputRef.current.value = "";
            if (folderInputRef.current) folderInputRef.current.value = "";
            return;
        }

        const updatedData = [...baseData, ...newSamples];
        
        // 排序：使用自然数字排序 (Natural Sort)
        const sorted = updatedData.sort((a, b) => {
             return a.fileName.localeCompare(b.fileName, undefined, { numeric: true, sensitivity: 'base' });
        });

        setTrainingData(sorted);
        
        if (trainingStrategy === 'new') {
            setTrainedModel(null);
        }

        saveDraftModel(sorted);
        
        if (!selectedSampleId && newSamples.length > 0) {
            setSelectedSampleId(newSamples[0].id);
        }

    } catch (err) {
        console.error(err);
        alert("导入失败");
    } finally {
        setIsProcessing(false);
        if (trainingInputRef.current) trainingInputRef.current.value = "";
        if (folderInputRef.current) folderInputRef.current.value = "";
    }
  };

  // 2. Training Action
  const handleTrainModel = () => {
    if (trainingData.length === 0) return;
    setIsProcessing(true);
    
    setTimeout(() => {
        try {
            const model = trainAndSaveModel(trainingData, modelName);
            setTrainedModel(model);
            setTrainingData([...model.samples]); 
            alert(`训练成功！\n\n当前生效模型: "${modelName}"\n包含样本数: ${model.samples.length}`);
        } catch (e) {
            console.error(e);
            alert("训练失败: " + (e as Error).message);
        } finally {
            setIsProcessing(false);
        }
    }, 100);
  };

  // 3. Test Data Upload (Detection)
  const handleTestUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files) return;
      handleProcessTestFiles(e.target.files);
      if (e.target) e.target.value = "";
  };

  const handleProcessTestFiles = (fileList: FileList) => {
      // Explicitly cast to File[]
      const files = (Array.from(fileList) as File[]).filter(f => f.name.toLowerCase().endsWith('.txt'));
      if (files.length === 0) {
          alert("未找到 .txt 文件");
          return;
      }
      setTestFiles(prev => {
          const combined = [...prev, ...files];
          // 排序：待测文件自然排序
          return combined.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));
      });
      setPredictions([]); 
      setDetectionStatus(DetectionStatus.IDLE);
      setSelectedPredictionIdx(null);
  }
  
  const handleRemoveTestFile = (e: React.MouseEvent, idx: number) => {
      e.stopPropagation();
      setTestFiles(prev => prev.filter((_, i) => i !== idx));
      setPredictions([]); 
      setDetectionStatus(DetectionStatus.IDLE);
  };
  
  const handleClearTestFiles = () => {
      if (testFiles.length === 0) return;
      setTestFiles([]);
      setPredictions([]);
      setDetectionStatus(DetectionStatus.IDLE);
      setSelectedPredictionIdx(null);
      if (testInputRef.current) testInputRef.current.value = "";
      if (testFolderInputRef.current) testFolderInputRef.current.value = "";
  }

  const handleStartDetection = async () => {
      if (!trainedModel) { 
          alert("未检测到已训练的模型。\n请先前往【模型训练】页面导入数据并完成训练，或【导入】已有模型。"); 
          return; 
      }
      setIsProcessing(true);
      const results: PredictionResult[] = [];
      for (const file of testFiles) {
          try {
              const { data } = await parseEEMFile(file);
              if (data.length > 0) {
                  const res = predictSample(data, trainedModel, file.name);
                  results.push(res);
              }
          } catch (e) { console.error(e); }
      }
      // 排序：结果自然排序
      results.sort((a, b) => a.fileName.localeCompare(b.fileName, undefined, { numeric: true, sensitivity: 'base' }));
      
      setPredictions(results);
      if (results.length > 0) {
          setDetectionStatus(DetectionStatus.PREDICTED);
          setSelectedPredictionIdx(0);
      }
      setIsProcessing(false);
  };

  const handleDownloadCSV = () => {
      if (predictions.length === 0) return;
      
      const headers = ["文件名", "预测分类", "预测浓度/比例", "置信度"];
      const rows = predictions.map(p => [
          p.fileName,
          p.predictedLabel,
          p.predictedRatio.toFixed(4),
          (p.confidence).toFixed(2) + "%"
      ]);
      
      const csvContent = "\uFEFF" + [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `detect_results_${new Date().toISOString().slice(0,10)}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
  };

  // ================= Derived State =================
  const uniqueLabels = useMemo(() => Array.from(new Set(trainingData.map(d => d.label))), [trainingData]);
  
  const chartData = useMemo(() => {
      if (trainedModel) return trainedModel.samples.map(s => ({ ...s, x: s.x || 0, y: s.y || 0 }));
      return trainingData.map(s => ({ ...s, x: 0, y: 0 }));
  }, [trainedModel, trainingData]);

  const selectedSample = trainingData.find(s => s.id === selectedSampleId);
  const currentPrediction = selectedPredictionIdx !== null ? predictions[selectedPredictionIdx] : null;
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#6366f1'];

  const isModelReady = trainedModel !== null;
  const modelStatusText = isModelReady ? "模型已就绪" : (trainingData.length > 0 ? "需重新训练" : "等待数据");
  const modelStatusColor = isModelReady ? "text-green-400" : (trainingData.length > 0 ? "text-orange-400" : "text-slate-500");
  const modelStatusDot = isModelReady ? "bg-green-400" : (trainingData.length > 0 ? "bg-orange-400 animate-pulse" : "bg-slate-500");

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">
      
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col shrink-0 z-20 shadow-xl">
        <div className="p-6 border-b border-slate-700">
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <BrainCircuit className="text-cyan-400" />
            FluorDetect
          </h1>
          <p className="text-xs mt-2 text-slate-400">工业废水荧光分析平台</p>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <SidebarItem icon={<BarChart3 size={20} />} label="模型训练" subLabel="Training" active={currentView === AppView.TRAINING} onClick={() => setCurrentView(AppView.TRAINING)} />
          <SidebarItem icon={<Microscope size={20} />} label="水样检测" subLabel="Detection" active={currentView === AppView.DETECTION} onClick={() => setCurrentView(AppView.DETECTION)} />
        </nav>
        <div className="p-4 bg-slate-800 text-xs border-t border-slate-700">
            <div className={`font-bold flex items-center gap-2 ${modelStatusColor}`}>
                <div className={`w-2 h-2 rounded-full ${modelStatusDot}`}></div>
                {modelStatusText}
            </div>
            {isModelReady && (
                <div className="mt-2 space-y-1">
                    <div className="text-slate-400 truncate font-mono" title={trainedModel?.name}>当前: {trainedModel?.name}</div>
                    <div className="text-slate-500">样本数: {trainedModel?.samples.length}</div>
                </div>
            )}
        </div>
      </aside>

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* VIEW: TRAINING */}
        {currentView === AppView.TRAINING && (
          <div className="flex-1 flex flex-col p-6 gap-6 h-full overflow-hidden">
            {/* Header Area */}
            <header className="flex flex-wrap justify-between items-center gap-4 shrink-0 bg-white p-4 rounded-xl shadow-sm border border-slate-200">
              <div className="flex flex-col gap-1">
                 <h2 className="text-xl font-bold text-slate-800">模型训练工作台</h2>
                 <div className="text-xs text-slate-500">导入荧光数据，构建识别模型</div>
              </div>
              
              <div className="flex items-center gap-3">
                 {/* Model Management Group */}
                 <div className="flex items-center gap-2 bg-slate-50 p-1.5 rounded-lg border border-slate-200">
                    <input 
                        type="text" 
                        value={modelName}
                        onChange={(e) => setModelName(e.target.value)}
                        placeholder="输入模型名称..."
                        className="text-sm px-2 py-1.5 border border-slate-300 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500 w-32"
                    />
                    
                    <button 
                        onClick={handleExportModel}
                        disabled={!trainedModel}
                        className="p-2 text-slate-600 hover:text-indigo-600 hover:bg-white rounded border border-transparent hover:border-slate-200 transition"
                        title="导出模型 (保存到本地)"
                    >
                        <FileDown size={18} />
                    </button>
                    
                    <button 
                        onClick={() => importModelInputRef.current?.click()}
                        className="p-2 text-slate-600 hover:text-indigo-600 hover:bg-white rounded border border-transparent hover:border-slate-200 transition"
                        title="导入模型 (从本地加载)"
                    >
                        <FileUp size={18} />
                    </button>
                    <input type="file" ref={importModelInputRef} onChange={handleImportModel} accept=".json" className="hidden" />
                 </div>

                 <div className="w-px h-8 bg-slate-200 mx-1"></div>

                 {/* Strategy Toggle */}
                 <div className="bg-slate-100 p-1 rounded-lg flex text-xs">
                    <button 
                        onClick={() => setTrainingStrategy('append')}
                        className={`px-3 py-1.5 rounded-md flex items-center gap-1 transition ${trainingStrategy === 'append' ? 'bg-white text-indigo-700 shadow-sm font-bold' : 'text-slate-500 hover:text-slate-700'}`}
                        title="保留现有数据，添加新上传的文件"
                    >
                        <Database size={14}/> 增量
                    </button>
                    <button 
                        onClick={() => setTrainingStrategy('new')}
                        className={`px-3 py-1.5 rounded-md flex items-center gap-1 transition ${trainingStrategy === 'new' ? 'bg-white text-indigo-700 shadow-sm font-bold' : 'text-slate-500 hover:text-slate-700'}`}
                        title="清空现有数据，重新开始"
                    >
                        <Eraser size={14}/> 全新
                    </button>
                 </div>

                 {/* Train Button */}
                 <button 
                    onClick={handleTrainModel} 
                    disabled={trainingData.length === 0 || isProcessing} 
                    className={`flex items-center gap-2 px-5 py-2.5 text-sm font-bold text-white rounded-lg shadow-md transition-all 
                      ${trainingData.length === 0 ? 'bg-slate-300 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-lg active:scale-95'}
                    `}
                  >
                    {isProcessing ? <RefreshCw className="animate-spin" size={16}/> : <Play size={16} fill="currentColor"/>}
                    开始训练
                  </button>
              </div>
            </header>

            <div className="flex-1 grid grid-cols-12 gap-6 overflow-hidden">
              {/* Import & List */}
              <div className="col-span-3 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col overflow-hidden">
                <div className="p-4 bg-slate-50 border-b border-slate-100">
                    <div className="flex justify-between items-center mb-3">
                        <h3 className="font-semibold text-slate-700 flex items-center gap-2 text-sm">
                            <UploadCloud size={16}/> 数据导入
                        </h3>
                        {/* Clear Data Button */}
                        {trainingData.length > 0 && (
                            <button 
                                onClick={handleClearTrainingData}
                                className="text-slate-400 hover:text-red-500 transition p-1.5 rounded hover:bg-red-50 border border-transparent hover:border-red-100"
                                title="清空列表 (不会删除已训练模型)"
                            >
                                <Trash2 size={16} />
                            </button>
                        )}
                    </div>
                    
                    <div className="flex gap-2">
                        <div className="flex-1">
                            <input type="file" multiple accept=".txt" ref={trainingInputRef} onChange={handleTrainingUpload} className="hidden" id="f-up"/>
                            <label htmlFor="f-up" className={`flex flex-col items-center justify-center py-3 border border-dashed border-blue-300 bg-blue-50 text-blue-700 rounded hover:bg-blue-100 cursor-pointer text-xs transition ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}><FileText size={16} className="mb-1"/> 选择文件</label>
                        </div>
                        <div className="flex-1">
                            <input type="file" {...{webkitdirectory:"", directory:""} as any} ref={folderInputRef} onChange={handleTrainingUpload} className="hidden" id="d-up"/>
                            <label htmlFor="d-up" className={`flex flex-col items-center justify-center py-3 border border-dashed border-slate-300 bg-slate-50 text-slate-600 rounded hover:bg-slate-100 cursor-pointer text-xs transition ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}><FolderUp size={16} className="mb-1"/> 文件夹</label>
                        </div>
                    </div>
                    {trainingStrategy === 'append' && <div className="text-[10px] text-slate-400 mt-2 text-center">当前模式：新上传数据将添加到现有列表</div>}
                    {trainingStrategy === 'new' && <div className="text-[10px] text-orange-400 mt-2 text-center">当前模式：上传时将清空现有数据！</div>}
                </div>

                <div className="p-2 px-4 bg-white border-b border-slate-100 flex justify-between items-center">
                    <span className="text-xs font-bold text-slate-500 uppercase flex items-center gap-1"><List size={12}/> 已导入文档 ({trainingData.length})</span>
                </div>

                <div className="flex-1 overflow-y-auto p-2 space-y-1 bg-slate-50/50">
                    {trainingData.length === 0 ? (
                        <div className="h-32 flex flex-col items-center justify-center text-slate-400 text-xs">暂无数据</div>
                    ) : (
                        trainingData.map(sample => (
                            <div 
                                key={sample.id} 
                                onClick={() => !isProcessing && setSelectedSampleId(sample.id)}
                                className={`group flex justify-between items-center p-2 rounded border text-xs cursor-pointer transition relative 
                                  ${selectedSampleId === sample.id ? 'bg-blue-50 border-blue-200 shadow-sm' : 'bg-white border-slate-100 hover:border-blue-200'}
                                  ${isProcessing ? 'pointer-events-none opacity-50' : ''}
                                `}
                            >
                                <div className="flex items-center gap-2 truncate flex-1 pr-6" title={sample.fileName}>
                                    <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${selectedSampleId === sample.id ? 'bg-blue-500' : 'bg-slate-300'}`}></div>
                                    <span className={`truncate font-medium ${selectedSampleId === sample.id ? 'text-blue-700' : 'text-slate-600'}`}>{sample.fileName}</span>
                                </div>
                                <button 
                                    type="button" 
                                    onClick={(e) => handleRemoveTrainingSample(e, sample.id)} 
                                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-full text-slate-300 hover:text-red-500 hover:bg-red-50 cursor-pointer transition z-10"
                                    title="删除此样本"
                                >
                                    <XCircle size={14} />
                                </button>
                            </div>
                        ))
                    )}
                </div>
              </div>

              {/* Charts & Table */}
              <div className="col-span-9 flex flex-col gap-6 overflow-hidden h-full">
                <div className="h-[380px] grid grid-cols-2 gap-6 shrink-0">
                    {/* PCA Chart Section */}
                    <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 flex flex-col">
                        <div className="flex justify-between items-center mb-2">
                             <h3 className="font-semibold text-slate-700 text-sm flex items-center gap-2"><Activity size={16} className="text-indigo-500"/> PCA 特征空间</h3>
                        </div>
                        <div className="flex-1 min-h-0 relative bg-slate-50/30 rounded border border-slate-100">
                             {isModelReady ? (
                                 <ResponsiveContainer width="100%" height="100%">
                                    <ReScatter margin={{top:20, right:20, bottom:20, left:0}}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="3 3" />
                                        <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
                                        <XAxis type="number" dataKey="x" name="PC1" tick={{fontSize:10}} unit="" stroke="#94a3b8" />
                                        <YAxis type="number" dataKey="y" name="PC2" tick={{fontSize:10}} unit="" stroke="#94a3b8" />
                                        <Tooltip cursor={{strokeDasharray:'3 3'}} content={({active, payload}) => {
                                            if(active && payload && payload.length) {
                                                const d = payload[0].payload;
                                                return (
                                                  <div className="bg-white/90 backdrop-blur p-2 border border-slate-200 shadow-lg text-xs rounded z-50">
                                                    <div className="font-bold text-slate-800 mb-1">{d.fileName}</div>
                                                    <div className="text-slate-500">类型: <span className="text-slate-800">{d.label}</span></div>
                                                    <div className="text-slate-500">浓度: <span className="text-slate-800">{d.ratio}%</span></div>
                                                  </div>
                                                );
                                            }
                                            return null;
                                        }}/>
                                        <Legend wrapperStyle={{fontSize:'10px'}} />
                                        {uniqueLabels.map((l, i) => (
                                            <Scatter key={l} name={l} data={chartData.filter(d=>d.label===l)} fill={colors[i%colors.length]} />
                                        ))}
                                    </ReScatter>
                                 </ResponsiveContainer>
                             ) : (
                                 <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 bg-slate-50">
                                     {trainingData.length > 0 ? (
                                        <>
                                            <AlertCircle size={32} className="mb-2 text-orange-400 animate-bounce"/>
                                            <span className="text-sm font-medium text-slate-600">数据已更新</span>
                                            <span className="text-xs mt-1">请点击右上角 <span className="font-bold text-indigo-600">开始模型训练</span> 以生成图表</span>
                                        </>
                                     ) : (
                                        <>
                                            <BarChart3 size={32} className="mb-2 opacity-20"/>
                                            <span className="text-xs">请导入数据并开始训练</span>
                                        </>
                                     )}
                                 </div>
                             )}
                        </div>
                    </div>

                    {/* Heatmap Section */}
                    <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 flex flex-col">
                         <div className="flex justify-between items-center mb-2">
                            <h3 className="font-semibold text-slate-700 text-sm flex items-center gap-2"><Microscope size={16} className="text-pink-500"/> 荧光图谱预览</h3>
                            {selectedSample && <span className="text-xs text-slate-400 truncate max-w-[150px]">{selectedSample.fileName}</span>}
                         </div>
                         <div className="flex-1 bg-slate-50 rounded border border-slate-100 flex items-center justify-center overflow-hidden relative">
                            {selectedSample && selectedSample.data.length > 0 ? (
                                <EEMHeatmap data={selectedSample.data} width={380} height={300} />
                            ) : (
                                <span className="text-slate-400 text-xs">请选择左侧文件查看图谱</span>
                            )}
                         </div>
                    </div>
                </div>

                {/* Features Table */}
                <div className="flex-1 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col overflow-hidden min-h-0">
                    <div className="p-3 px-4 border-b border-slate-100 bg-slate-50 shrink-0">
                        <h3 className="font-semibold text-slate-700 text-sm">训练集特征详情</h3>
                    </div>
                    <div className="flex-1 overflow-auto">
                        <table className="w-full text-xs text-left text-slate-600">
                            <thead className="bg-white text-slate-500 font-semibold sticky top-0 shadow-sm z-10">
                                <tr>
                                    <th className="px-4 py-2 border-b">文件名</th>
                                    <th className="px-4 py-2 border-b">标签</th>
                                    <th className="px-4 py-2 border-b">Max强度</th>
                                    <th className="px-4 py-2 border-b">Peak(Ex/Em)</th>
                                    <th className="px-4 py-2 border-b">Entropy</th>
                                    <th className="px-4 py-2 border-b">Sparsity</th>
                                </tr>
                            </thead>
                            <tbody>
                                {trainingData.map(s => (
                                    <tr key={s.id} className={`border-b border-slate-50 hover:bg-blue-50 cursor-pointer transition ${selectedSampleId === s.id ? 'bg-blue-50' : ''}`} onClick={() => setSelectedSampleId(s.id)}>
                                        <td className="px-4 py-2 font-medium text-slate-900 truncate max-w-[180px]" title={s.fileName}>{s.fileName}</td>
                                        <td className="px-4 py-2"><span className="bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">{s.label}</span></td>
                                        <td className="px-4 py-2">{s.features.maxIntensity.toFixed(1)}</td>
                                        <td className="px-4 py-2 font-mono text-slate-400">{s.features.peakEx}/{s.features.peakEm}</td>
                                        <td className="px-4 py-2 font-mono text-indigo-600">{s.features.entropy?.toFixed(3)}</td>
                                        <td className="px-4 py-2 text-slate-400">{(s.features.sparsity*100).toFixed(1)}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    {/* UPDATED: Footer with Category Count */}
                    <div className="p-2 px-4 bg-slate-100 border-t border-slate-200 text-xs flex gap-6 shrink-0">
                        <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-blue-500"></span><span className="text-slate-500">样本总数:</span><span className="font-bold text-slate-800">{trainingData.length}</span>
                        </div>
                        <div className="flex items-center gap-2">
                             <span className="w-2 h-2 rounded-full bg-purple-500"></span><span className="text-slate-500">类别数量:</span><span className="font-bold text-slate-800">{uniqueLabels.length}</span>
                        </div>
                    </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW: DETECTION */}
        {currentView === AppView.DETECTION && (
          <div className="flex-1 flex flex-col p-6 gap-6 h-full overflow-hidden">
             
             {/* DETECTION HEADER */}
             <header className="flex justify-between items-center shrink-0">
                <div><h2 className="text-2xl font-bold text-slate-800">水样检测</h2></div>
                <div className="flex gap-4 items-center">
                    {/* Current Active Model Banner */}
                    <div className={`px-4 py-2 rounded-lg border flex items-center gap-3 shadow-sm ${trainedModel ? 'bg-indigo-50 border-indigo-200 text-indigo-900' : 'bg-slate-100 border-slate-200 text-slate-500'}`}>
                        {trainedModel ? (
                            <>
                                <CheckCircle size={18} className="text-green-600"/>
                                <div>
                                    <div className="text-xs text-indigo-500 font-semibold uppercase tracking-wider">当前使用模型 (Active Model)</div>
                                    <div className="font-bold text-sm flex items-center gap-2">
                                        {trainedModel.name || "Default Model"}
                                        <span className="text-[10px] bg-white px-1.5 py-0.5 rounded border border-indigo-100 text-slate-500 font-normal">
                                            {trainedModel.timestamp ? new Date(trainedModel.timestamp).toLocaleTimeString() : 'N/A'}
                                        </span>
                                    </div>
                                </div>
                            </>
                        ) : (
                            <>
                                <AlertTriangle size={18} className="text-orange-400"/>
                                <div className="text-xs font-medium">未加载模型 (请先训练或导入)</div>
                            </>
                        )}
                    </div>
                    
                    <button onClick={handleStartDetection} disabled={isProcessing || testFiles.length===0 || !trainedModel} className="bg-cyan-600 text-white px-6 py-2.5 rounded-lg text-sm font-bold shadow hover:bg-cyan-700 disabled:bg-slate-300 flex items-center gap-2 h-full">
                        {isProcessing ? <RefreshCw className="animate-spin" size={16}/> : <Play size={16} fill="currentColor"/>}
                        {isProcessing ? "检测中..." : "开始检测"}
                    </button>
                </div>
             </header>

             <div className="flex-1 grid grid-cols-12 gap-6 overflow-hidden">
                {/* Upload List (Left) */}
                <div className="col-span-3 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col overflow-hidden">
                    <div className="p-4 bg-slate-50 border-b border-slate-100">
                        <div className="flex justify-between items-center mb-3">
                            <h3 className="font-semibold text-slate-700 flex items-center gap-2 text-sm">
                                <UploadCloud size={16}/> 上传待测文件
                            </h3>
                            {testFiles.length > 0 && (
                                <button 
                                    onClick={handleClearTestFiles}
                                    className="text-slate-400 hover:text-red-500 transition p-1.5 rounded hover:bg-red-50 border border-transparent hover:border-red-100"
                                    title="清空列表"
                                >
                                    <Trash2 size={16} />
                                </button>
                            )}
                        </div>
                        <div className="flex gap-2">
                            <div className="flex-1">
                                <input type="file" multiple accept=".txt" ref={testInputRef} onChange={handleTestUpload} className="hidden" id="tf-up"/>
                                <label htmlFor="tf-up" className="flex flex-col items-center justify-center py-3 border border-dashed border-blue-300 bg-blue-50 text-blue-700 rounded hover:bg-blue-100 cursor-pointer text-xs transition"><FileText size={16} className="mb-1"/> 选择文件</label>
                            </div>
                            <div className="flex-1">
                                <input type="file" {...{webkitdirectory:"", directory:""} as any} ref={testFolderInputRef} onChange={handleTestUpload} className="hidden" id="td-up"/>
                                <label htmlFor="td-up" className="flex flex-col items-center justify-center py-3 border border-dashed border-slate-300 bg-slate-50 text-slate-600 rounded hover:bg-slate-100 cursor-pointer text-xs transition"><FolderUp size={16} className="mb-1"/> 文件夹</label>
                            </div>
                        </div>
                    </div>
                    <div className="p-2 px-4 bg-white border-b border-slate-100 flex justify-between items-center">
                        <span className="text-xs font-bold text-slate-500 uppercase flex items-center gap-1"><List size={12}/> 待测文件 ({testFiles.length})</span>
                    </div>
                    <div className="flex-1 overflow-y-auto p-2 space-y-1 bg-slate-50/50">
                        {testFiles.map((f, i) => {
                            const pred = predictions.find(p => p.fileName === f.name);
                            return (
                                <div key={i} onClick={() => pred && setSelectedPredictionIdx(predictions.indexOf(pred))} className={`group flex justify-between items-center p-2 rounded border text-xs transition relative cursor-pointer ${pred && selectedPredictionIdx === predictions.indexOf(pred) ? 'bg-blue-50 border-blue-200' : 'bg-white border-slate-100'}`}>
                                    <div className="flex items-center gap-2 truncate flex-1 pr-6">
                                        <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${pred ? 'bg-green-500' : 'bg-slate-300'}`}></div>
                                        <span className={`truncate font-medium ${pred ? 'text-slate-800' : 'text-slate-500'}`}>{f.name}</span>
                                    </div>
                                    <button type="button" onClick={(e) => handleRemoveTestFile(e, i)} className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-full text-slate-300 hover:text-red-500 hover:bg-red-50 cursor-pointer transition z-10"><XCircle size={14} /></button>
                                </div>
                            );
                        })}
                    </div>
                </div>

                <div className="col-span-9 flex flex-col gap-4 overflow-hidden h-full">
                    {/* Top Section: Detailed Analysis - UPDATED LAYOUT */}
                    <div className="h-[380px] shrink-0 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col overflow-hidden"> 
                        {/* New Header Bar for Specific Details */}
                        <div className="px-6 py-3 border-b border-slate-100 bg-slate-50 flex items-center justify-between shrink-0">
                            <h3 className="font-semibold text-slate-700 text-sm flex items-center gap-2">
                                <Activity size={16} className="text-indigo-600"/> 具体详情
                            </h3>
                        </div>
                        
                        <div className="p-6 overflow-y-auto flex-1">
                            {currentPrediction ? (
                                <div className="animate-in fade-in zoom-in-95 duration-200">
                                    <div className="flex justify-between items-center mb-6 border-b border-slate-100 pb-4">
                                        <h3 className="text-xl font-bold text-slate-800">{currentPrediction.fileName}</h3>
                                        <div className="flex items-center gap-4">
                                            <span className="text-4xl font-extrabold text-slate-700">检测值:</span>
                                            <div className="text-4xl font-extrabold text-indigo-700">{currentPrediction.predictedLabel}</div>
                                        </div>
                                    </div>
                                    
                                    <div className="grid grid-cols-2 gap-6 mb-6">
                                        <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                                            <span className="text-xs text-slate-400 uppercase font-bold">浓度/比例</span>
                                            <div className="text-2xl font-bold text-slate-800 mt-1">{currentPrediction.predictedRatio.toFixed(2)}%</div>
                                        </div>
                                        <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                                            <span className="text-xs text-slate-400 uppercase font-bold">置信度</span>
                                            <div className="text-2xl font-bold text-slate-800 mt-1">{currentPrediction.confidence.toFixed(1)}%</div>
                                        </div>
                                    </div>

                                    <div>
                                        <h4 className="font-semibold text-slate-700 mb-3 text-sm flex items-center gap-2"><BrainCircuit size={16}/> 最近邻匹配 (KNN)</h4>
                                        <div className="space-y-2">
                                            {currentPrediction.nearestNeighbors.map((n, i) => (
                                                <div key={i} className="flex items-center justify-between p-3 bg-white border border-slate-100 rounded hover:border-blue-200 transition">
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-bold">{i+1}</div>
                                                        <div>
                                                            <div className="text-sm font-medium text-slate-800">{n.fileName}</div>
                                                            <div className="text-xs text-slate-400">Label: {n.label} | Ratio: {n.ratio}</div>
                                                        </div>
                                                    </div>
                                                    <div className="text-xs font-mono text-slate-500">Dist: {currentPrediction.distances[i].toFixed(4)}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center text-slate-300">
                                    <Microscope size={64} className="mb-4 opacity-10"/>
                                    <p className="text-sm mb-1">{isModelReady ? "请上传文件并点击“开始检测”" : "需要先在【模型训练】页面训练模型"}</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Bottom Section: Batch Results Table - UPDATED VISUALS WITH EMPTY STATE */}
                    <div className="flex-1 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col min-h-0">
                        <div className="px-6 py-3 border-b border-slate-100 bg-slate-50 flex justify-between items-center shrink-0">
                            <h3 className="font-semibold text-slate-700 text-sm flex items-center gap-2"><Layers size={16} className="text-blue-600"/> 批量检测结果 {predictions.length > 0 && `(${predictions.length})`}</h3>
                            <button 
                                onClick={handleDownloadCSV} 
                                disabled={predictions.length === 0}
                                className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded border transition
                                    ${predictions.length > 0 
                                        ? 'text-indigo-600 hover:text-indigo-800 bg-white border-slate-200 hover:border-indigo-300' 
                                        : 'text-slate-400 bg-slate-50 border-transparent cursor-not-allowed'}`}
                            >
                                <Download size={12} /> 导出 CSV
                            </button>
                        </div>
                        <div className="flex-1 overflow-auto bg-slate-50/30 relative">
                            {predictions.length > 0 ? (
                                <table className="w-full text-xs text-left text-slate-600">
                                    <thead className="bg-white text-slate-500 font-semibold sticky top-0 shadow-sm z-10">
                                        <tr>
                                            <th className="px-6 py-3 border-b">文件名</th>
                                            <th className="px-6 py-3 border-b">分类结果</th>
                                            <th className="px-6 py-3 border-b">浓度/比例</th>
                                            <th className="px-6 py-3 border-b">置信度</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {predictions.map((p, i) => (
                                            <tr 
                                                key={i} 
                                                onClick={() => setSelectedPredictionIdx(i)}
                                                className={`border-b border-slate-50 hover:bg-blue-50 cursor-pointer transition ${selectedPredictionIdx === i ? 'bg-blue-50' : ''}`}
                                            >
                                                <td className="px-6 py-3 font-medium text-slate-900 truncate max-w-[200px]" title={p.fileName}>{p.fileName}</td>
                                                <td className="px-6 py-3"><span className="bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200 font-bold text-slate-700">{p.predictedLabel}</span></td>
                                                <td className="px-6 py-3">{p.predictedRatio.toFixed(2)}%</td>
                                                <td className="px-6 py-3 text-slate-400">{p.confidence.toFixed(1)}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            ) : (
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-300">
                                    <List size={48} className="mb-3 opacity-20"/>
                                    <p className="text-sm font-medium">暂无数据</p>
                                    <p className="text-xs mt-1">批量检测结果将显示在这里</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
             </div>
          </div>
        )}
      </main>
    </div>
  );
}

function SidebarItem({ icon, label, subLabel, active, onClick }: any) {
  return (
    <button onClick={onClick} className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition ${active ? 'bg-slate-800 text-white border-l-4 border-cyan-400' : 'text-slate-400 hover:bg-slate-800'}`}>
      <div className={active ? 'text-cyan-400' : ''}>{icon}</div>
      <div><div className="font-medium text-sm">{label}</div><div className="text-[10px] opacity-60 uppercase">{subLabel}</div></div>
    </button>
  );
}
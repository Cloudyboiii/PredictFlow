"use client";
import { useState, useRef, useCallback, useEffect } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend
} from "recharts";
import { uploadCSV, trainModels, predict, deleteDataset, batchPredict, profileDataset, explainPrediction } from "@/lib/api";

/* ---- Types ---- */
interface ColInfo { name: string; type: string; null_count: number; unique_count: number; sample_values: any[]; }
interface Dataset { filename: string; row_count: number; column_count: number; columns: ColInfo[]; }
interface Metrics { accuracy?: number; precision?: number; recall?: number; f1_score?: number; roc_auc?: number; cv_mean: number | null; cv_std: number | null; confusion_matrix?: number[][]; roc_curve?: { fpr: number; tpr: number }[]; mae?: number; rmse?: number; r2_score?: number; mape?: number; residual_plot?: {actual: number, predicted: number}[]; }
interface ModelResult { name: string; metrics: Metrics; feature_importance: { feature: string; importance: number }[]; is_best: boolean; error?: string; }
interface TrainResult { models: ModelResult[]; best_model: string; target_col: string; label_classes: string[]; num_classes: number; feature_names: string[]; training_samples: number; test_samples: number; task_type: "classification" | "regression"; }

interface ProfileData {
  columns: any[];
  class_balance: any[];
  correlation_matrix: any[];
  overall_health_score: number;
  row_count: number;
  col_count: number;
}

const MODEL_COLORS: Record<string, string> = {
  "Logistic Regression": "#6366f1",
  "Linear Regression": "#6366f1",
  "Random Forest": "#059669",
  "Random Forest Regressor": "#059669",
  "XGBoost": "#d97706",
  "XGBRegressor": "#d97706",
  "Neural Network": "#dc2626",
};

const METRICS_LABELS: Record<string, string> = {
  accuracy: "Accuracy", precision: "Precision", recall: "Recall",
  f1_score: "F1 Score", roc_auc: "ROC-AUC",
};

export default function Home() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [target, setTarget] = useState("");
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"metrics" | "roc" | "features" | "predict" | "explain">("metrics");
  const [uploading, setUploading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [predictInputs, setPredictInputs] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const batchFileRef = useRef<HTMLInputElement>(null);

  // Feature 2: Profiling
  const [profileData, setProfileData] = useState<ProfileData | null>(null);
  const [showProfile, setShowProfile] = useState(true);
  const [profiling, setProfiling] = useState(false);

  // Feature 1: Batch predict
  const [batchPredicting, setBatchPredicting] = useState(false);
  const [batchMsg, setBatchMsg] = useState("");

  // Feature 3: SHAP Explain
  const [explainData, setExplainData] = useState<any>(null);
  const [explaining, setExplaining] = useState(false);
  const [explainingRow, setExplainingRow] = useState(0);

  /* Upload */
  const handleUpload = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".csv")) { setError("Only CSV files accepted."); return; }
    setUploading(true); setError(null); setTrainResult(null); setTarget(""); setProfileData(null);
    try {
      const res = await uploadCSV(file);
      setDataset(res);
      const cols: string[] = [];
      res.columns.forEach((c: ColInfo) => {
        const nameLower = c.name.toLowerCase();
        const isId = nameLower.includes("id") || c.unique_count === res.row_count;
        if (!isId) cols.push(c.name);
      });
      setSelectedColumns(cols);
      
      // Load profile
      setProfiling(true);
      try {
        const prof = await profileDataset();
        setProfileData(prof);
        setShowProfile(true);
      } catch (pe) {
        console.error("Profiling failed", pe);
      }
      setProfiling(false);
      
    } catch (e: any) { setError(e.message); }
    finally { setUploading(false); }
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleUpload(f);
  };

  /* Train */
  const handleTrain = async () => {
    if (!target) { setError("Please select a target column."); return; }
    setTraining(true); setError(null); setTrainResult(null); setPrediction(null); setExplainData(null);
    try {
      const featureCols = selectedColumns.filter(c => c !== target);
      const res = await trainModels(target, featureCols);
      setTrainResult(res);
      setSelectedModel(res.best_model);
      setActiveTab("metrics");
      // Init predict inputs
      const inputs: Record<string, string> = {};
      res.feature_names.forEach((f: string) => { inputs[f] = ""; });
      setPredictInputs(inputs);
    } catch (e: any) { setError(e.message); }
    finally { setTraining(false); }
  };

  /* Predict */
  const handlePredict = async () => {
    setPredicting(true); setError(null); setPrediction(null);
    try {
        const features: Record<string, any> = {};
      Object.entries(predictInputs).forEach(([k, v]) => {
          const colInfo = dataset?.columns.find(c => c.name === k);
          if (colInfo?.type === 'numeric' || !isNaN(parseFloat(v))) {
              features[k] = parseFloat(v) || 0;
          } else {
              features[k] = v;
          }
      });
      const res = await predict(features, selectedModel);
      setPrediction({ ...res, featuresUsed: features });
    } catch (e: any) { setError(e.message); }
    finally { setPredicting(false); }
  };
  
  /* Batch Predict */
  const handleBatchPredict = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".csv")) { setError("Only CSV files accepted."); return; }
    setBatchPredicting(true); setError(null); setBatchMsg(`Running predictions on batch file...`);
    try {
      const blob = await batchPredict(file, selectedModel);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "predictions.csv";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setBatchMsg(`Done! predictions.csv downloaded successfully.`);
    } catch (e: any) { setError(e.message); setBatchMsg(""); }
    finally { setBatchPredicting(false); }
  };

  /* Explain specific prediction */
  const handleExplainPrediction = async (features: any) => {
    setActiveTab("explain");
    setExplaining(true); setError(null); setExplainData(null);
    try {
      const res = await explainPrediction(selectedModel, -1, features);
      setExplainData(res);
    } catch (e: any) { setError(e.message); }
    finally { setExplaining(false); }
  };
  
  const handleExplainRow = async (rowIdx: number) => {
    setExplaining(true); setError(null); setExplainData(null); setExplainingRow(rowIdx);
    try {
      const res = await explainPrediction(selectedModel, rowIdx);
      setExplainData(res);
    } catch (e: any) { setError(e.message); }
    finally { setExplaining(false); }
  };

  /* Clear */
  const handleClear = async () => {
    await deleteDataset().catch(() => {});
    setDataset(null); setTrainResult(null); setTarget(""); setError(null); setPrediction(null); setProfileData(null); setExplainData(null);
  };

  const currentModel = trainResult?.models.find(m => m.name === selectedModel);

  /* Comparison data for bar chart */
  const comparisonData = trainResult
    ? Object.entries(METRICS_LABELS).map(([key, label]) => {
        const row: any = { metric: label };
        trainResult.models.forEach(m => {
          if (!m.error) row[m.name] = (m.metrics as any)[key] ?? 0;
        });
        return row;
      })
    : [];

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white border-b border-border sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-brand flex items-center justify-center">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <div>
              <h1 className="text-[15px] font-semibold text-text">PredictFlow</h1>
              <p className="text-[11px] text-text-muted -mt-0.5">AutoML Training & Evaluation</p>
            </div>
          </div>
          {dataset && (
            <button onClick={handleClear} className="text-[12px] px-3 h-8 rounded-lg border border-border hover:border-red-300 text-text-muted hover:text-red-500 transition-colors">
              Clear & Reset
            </button>
          )}
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {error && (
          <div className="mb-4 px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-[13px]">
            {error}
          </div>
        )}

        {/* Upload Section */}
        {!dataset ? (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-text mb-2">AutoML in seconds</h2>
              <p className="text-text-secondary text-[14px]">Upload a CSV, select your target column, and PredictFlow trains 4 ML models simultaneously — comparing them with full metrics, ROC curves, and feature importance.</p>
            </div>

            {/* Upload zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${dragging ? "border-brand bg-brand/[0.03]" : "border-border hover:border-brand/40 hover:bg-brand/[0.02]"}`}
            >
              {uploading ? (
                <div>
                  <div className="w-10 h-10 border-2 border-brand/30 border-t-brand rounded-full animate-spin mx-auto mb-3"/>
                  <p className="text-[13px] text-text-secondary">Processing CSV...</p>
                </div>
              ) : (
                <div>
                  <div className="w-12 h-12 rounded-xl bg-brand/[0.08] mx-auto mb-3 flex items-center justify-center">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                      <path d="M12 16V8m0 0l-3 3m3-3l3 3M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <p className="text-[14px] font-medium text-text mb-1">Drop a CSV file here</p>
                  <p className="text-[12px] text-text-muted">or click to browse — up to 25MB, 50K rows</p>
                </div>
              )}
              <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f); }}/>
            </div>

            {/* What it trains */}
            <div className="mt-6 grid grid-cols-2 gap-3">
              {["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"].map((name) => (
                <div key={name} className="bg-white rounded-xl border border-border p-3 flex items-center gap-2.5">
                  <div className="w-3 h-3 rounded-full" style={{ background: MODEL_COLORS[name] }}/>
                  <span className="text-[13px] text-text-secondary">{name}</span>
                </div>
              ))}
            </div>
            <p className="text-center text-[12px] text-text-muted mt-4">All 4 models trained simultaneously with cross-validation, ROC-AUC, confusion matrices, and feature importance</p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Dataset bar */}
            <div className="bg-white rounded-xl border border-border px-5 py-3 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-7 h-7 rounded-lg bg-success/[0.08] flex items-center justify-center">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                    <path d="M9 12l2 2 4-4M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="#059669" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div>
                  <p className="text-[13px] font-medium text-text">{dataset.filename}</p>
                  <p className="text-[11px] text-text-muted">{dataset.row_count.toLocaleString()} rows · {dataset.column_count} columns</p>
                </div>
              </div>
              <button onClick={() => fileRef.current?.click()} className="text-[12px] text-brand hover:underline">Replace</button>
              <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f); }}/>
            </div>
            
            {/* Profiling View */}
            {!trainResult && profiling && (
                <div className="p-5 text-center text-[13px] text-text-muted"><div className="w-6 h-6 border-2 border-brand/30 border-t-brand rounded-full animate-spin mx-auto mb-2"/> Profiling dataset...</div>
            )}
            
            {!trainResult && profileData && (
              <div className="bg-white rounded-xl border border-border overflow-hidden">
                <div className="px-5 py-4 flex items-center justify-between bg-surface cursor-pointer hover:bg-surface-muted transition-colors" onClick={() => setShowProfile(!showProfile)}>
                  <div>
                    <h3 className="text-[14px] font-semibold text-text">Dataset Overview</h3>
                    <p className="text-[12px] text-text-muted">Health score, distributions, and correlations</p>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className={`px-3 py-1 rounded-full text-[12px] font-bold ${profileData.overall_health_score >= 80 ? 'bg-green-100 text-green-700' : profileData.overall_health_score >= 60 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                      Health Score: {profileData.overall_health_score}/100
                    </div>
                    <span className="text-text-muted">{showProfile ? "▲" : "▼"}</span>
                  </div>
                </div>
                
                {showProfile && (
                  <div className="p-5 border-t border-border">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                      {/* Top Correlations */}
                      <div>
                        <h4 className="text-[13px] font-semibold text-text mb-2">Top Correlations (Pearson)</h4>
                        {profileData.correlation_matrix.length > 0 ? (
                            <ul className="space-y-1">
                                {profileData.correlation_matrix.slice(0, 5).map((corr, i) => (
                                    <li key={i} className="text-[12px] text-text-secondary">
                                        <span className="font-medium text-text">{corr.col1}</span> ↔ <span className="font-medium text-text">{corr.col2}</span>: {corr.coefficient.toFixed(2)}
                                        {Math.abs(corr.coefficient) > 0.7 && <span className="ml-2 text-[10px] bg-brand/10 text-brand px-1.5 py-0.5 rounded">Strong</span>}
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="text-[12px] text-text-muted">No significant correlations found.</p>
                        )}
                      </div>
                      {/* Class Balance (potential targets) */}
                      <div>
                        <h4 className="text-[13px] font-semibold text-text mb-2">Potential Targets Class Balance</h4>
                        {profileData.class_balance.length > 0 ? (
                            <div className="space-y-3">
                                {profileData.class_balance.slice(0, 3).map((cb, i) => (
                                    <div key={i}>
                                        <p className="text-[11px] font-medium text-text mb-1">{cb.column_name}</p>
                                        <div className="flex w-full h-2 rounded overflow-hidden">
                                            {cb.classes.map((cls: any, j: number) => (
                                                <div key={j} style={{ width: `${cls.percent}%`, backgroundColor: j % 2 === 0 ? '#6366f1' : '#cbd5e1' }} title={`${cls.label}: ${cls.percent}%`} />
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-[12px] text-text-muted">No suitable categorical targets detected.</p>
                        )}
                      </div>
                    </div>
                    
                    <h4 className="text-[13px] font-semibold text-text mb-3">Column Profiles</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                        {profileData.columns.map(c => (
                            <div key={c.name} className="border border-border rounded-lg p-3 bg-surface-muted/30">
                                <div className="flex justify-between items-start mb-2">
                                    <h5 className="text-[12px] font-semibold text-text truncate max-w-[150px]" title={c.name}>{c.name}</h5>
                                    <span className="text-[10px] text-text-muted bg-border/50 px-1.5 py-0.5 rounded">{c.col_type}</span>
                                </div>
                                
                                {c.null_count > 0 && <span className="inline-block mb-1 text-[10px] bg-orange-100 text-orange-700 px-1.5 py-0.5 rounded">Nulls: {c.null_count} ({c.null_percent}%)</span>}
                                {c.outlier_count > 0 && <span className="inline-block mb-1 ml-1 text-[10px] bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded">Outliers: {c.outlier_count}</span>}
                                
                                {c.col_type === 'numeric' && c.histogram && c.histogram.length > 0 && (
                                    <div className="h-16 mt-2">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={c.histogram}>
                                                <Tooltip contentStyle={{fontSize: 10, padding: '2px 4px'}} formatter={(val: any) => [val, 'Count']} />
                                                <Bar dataKey="count" fill="#94a3b8" />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}
                                {c.col_type === 'numeric' && (
                                    <div className="grid grid-cols-2 gap-1 text-[10px] text-text-muted mt-2">
                                        <span>Min: {c.min?.toFixed(2)}</span>
                                        <span>Max: {c.max?.toFixed(2)}</span>
                                        <span>Mean: {c.mean?.toFixed(2)}</span>
                                        <span>Med: {c.median?.toFixed(2)}</span>
                                    </div>
                                )}
                                
                                {c.col_type === 'categorical' && c.top_values && (
                                    <div className="mt-2 space-y-1">
                                        <p className="text-[10px] text-text-muted">Top values:</p>
                                        <div className="flex flex-wrap gap-1">
                                            {c.top_values.slice(0,3).map((v: any, i: number) => (
                                                <span key={i} className="text-[10px] bg-white border border-border px-1.5 py-0.5 rounded truncate max-w-[100px]" title={v.value}>{v.value}</span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Target selector + Train */}
            {!trainResult && (
              <div className="bg-white rounded-xl border border-border p-5">
                <div className="mb-6 border-b border-border pb-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-[14px] font-semibold text-text mb-1">Select Feature Columns</h3>
                      <p className="text-[12px] text-text-muted">Choose which columns to include in the training data.</p>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={() => setSelectedColumns(dataset.columns.map(c => c.name))} className="text-[12px] px-3 py-1 rounded border border-border text-text-secondary hover:bg-surface-muted">Select All</button>
                      <button onClick={() => setSelectedColumns([])} className="text-[12px] px-3 py-1 rounded border border-border text-text-secondary hover:bg-surface-muted">Deselect All</button>
                    </div>
                  </div>
                  <div className="max-h-64 overflow-y-auto border border-border rounded-lg">
                    <table className="w-full text-[12px] text-left">
                      <thead className="bg-surface sticky top-0 border-b border-border">
                        <tr>
                          <th className="px-3 py-2 font-medium text-text-secondary w-8"></th>
                          <th className="px-3 py-2 font-medium text-text-secondary">Column Name</th>
                          <th className="px-3 py-2 font-medium text-text-secondary">Type</th>
                          <th className="px-3 py-2 font-medium text-text-secondary text-right">Unique</th>
                          <th className="px-3 py-2 font-medium text-text-secondary text-right">Missing</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {dataset.columns.map(c => (
                          <tr key={c.name} className="hover:bg-surface-muted/50 transition-colors">
                            <td className="px-3 py-2">
                              <input type="checkbox" checked={selectedColumns.includes(c.name)} onChange={(e) => {
                                if (e.target.checked) setSelectedColumns(prev => [...prev, c.name]);
                                else setSelectedColumns(prev => prev.filter(n => n !== c.name));
                              }} className="rounded border-border text-brand focus:ring-brand"/>
                            </td>
                            <td className="px-3 py-2 font-medium text-text">{c.name}</td>
                            <td className="px-3 py-2 text-text-muted">{c.type}</td>
                            <td className="px-3 py-2 text-text-muted text-right">{c.unique_count}</td>
                            <td className="px-3 py-2 text-text-muted text-right">{c.null_count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <h3 className="text-[14px] font-semibold text-text mb-1">Select Target Column</h3>
                <p className="text-[12px] text-text-muted mb-4">Choose the column you want to predict (Classification or Regression).</p>
                <div className="flex gap-3">
                  <select value={target} onChange={(e) => setTarget(e.target.value)}
                    className="flex-1 h-10 px-3 rounded-lg border border-border text-[13px] text-text focus:outline-none focus:border-brand/40">
                    <option value="">-- Select target column --</option>
                    {dataset.columns.map(c => (
                      <option key={c.name} value={c.name}>{c.name} ({c.unique_count} unique vals)</option>
                    ))}
                  </select>
                  <button onClick={handleTrain} disabled={training || !target}
                    className="px-6 h-10 rounded-lg bg-brand hover:bg-brand-light text-white text-[13px] font-medium disabled:opacity-40 transition-colors min-w-[120px]">
                    {training ? (
                      <span className="flex items-center gap-2">
                        <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"/>
                        Training...
                      </span>
                    ) : "Train Models"}
                  </button>
                </div>
                {training && (
                  <div className="mt-4 p-3 rounded-lg bg-brand/[0.04] border border-brand/20">
                    <p className="text-[12px] text-brand font-medium">🔄 Training 4 models simultaneously...</p>
                    <p className="text-[11px] text-text-muted mt-1">Logistic Regression → Random Forest → XGBoost → Neural Network</p>
                  </div>
                )}
              </div>
            )}

            {/* Results */}
            {trainResult && (
              <>
                {/* Best model banner */}
                <div className="bg-white rounded-xl border border-brand/30 p-4 flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: (MODEL_COLORS[trainResult.best_model] || "#000") + "20" }}>
                    <span className="text-lg">🏆</span>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-0.5">
                      <p className="text-[13px] font-semibold text-text">Best Model: {trainResult.best_model}</p>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${trainResult.task_type === 'regression' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}`}>
                        {trainResult.task_type === 'regression' ? 'Regression' : 'Classification'}
                      </span>
                    </div>
                    <p className="text-[12px] text-text-muted">
                      {trainResult.task_type === 'regression' ? (
                        <>R² Score: {trainResult.models.find(m => m.name === trainResult.best_model)?.metrics.r2_score?.toFixed(4)} · MAE: {trainResult.models.find(m => m.name === trainResult.best_model)?.metrics.mae?.toFixed(4)}</>
                      ) : (
                        <>ROC-AUC: {trainResult.models.find(m => m.name === trainResult.best_model)?.metrics.roc_auc?.toFixed(4)} · Accuracy: {trainResult.models.find(m => m.name === trainResult.best_model)?.metrics.accuracy?.toFixed(4)}</>
                      )}
                      {' '}· {trainResult.training_samples} training / {trainResult.test_samples} test samples
                    </p>
                  </div>
                  <a href={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:10000"}/api/download/best`}
                    className="text-[12px] px-3 h-8 flex items-center justify-center rounded-lg border border-brand text-brand hover:bg-brand/[0.05] transition-colors" title="Download trained model as .pkl for deployment">
                    Download Model
                  </a>
                  <button onClick={() => { setTrainResult(null); setPrediction(null); setExplainData(null); }}
                    className="text-[12px] px-3 h-8 rounded-lg border border-border text-text-muted hover:text-brand hover:border-brand/30 transition-colors">
                    Retrain
                  </button>
                </div>

                {/* Model selector */}
                <div className="flex gap-2 flex-wrap">
                  {trainResult.models.filter(m => !m.error).map(m => (
                    <div key={m.name} className="flex">
                      <button onClick={() => { setSelectedModel(m.name); setExplainData(null); setPrediction(null); }}
                        className={`flex items-center gap-2 px-4 py-2 rounded-l-xl border border-r-0 text-[12px] font-medium transition-all ${selectedModel === m.name ? "border-brand bg-brand/[0.06] text-brand" : "border-border text-text-secondary hover:border-brand/30"}`}>
                        <span className="w-2.5 h-2.5 rounded-full" style={{ background: MODEL_COLORS[m.name] }}/>
                        {m.name}
                        {m.is_best && <span className="text-[10px] bg-brand/10 text-brand px-1.5 py-0.5 rounded">Best</span>}
                      </button>
                      <a href={`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:10000"}/api/download/${encodeURIComponent(m.name)}`}
                         className={`flex items-center px-2 border rounded-r-xl transition-colors ${selectedModel === m.name ? "border-brand bg-brand/[0.06] text-brand hover:bg-brand/10" : "border-border text-text-muted hover:text-text hover:bg-surface-muted"}`}
                         title="Download model (.pkl)">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      </a>
                    </div>
                  ))}
                </div>

                {currentModel && !currentModel.error && (
                  <div className="bg-white rounded-xl border border-border overflow-hidden">
                    {/* Tabs */}
                    <div className="flex border-b border-border">
                      {(["metrics", "roc", "features", "predict", "explain"] as const).map(tab => (
                        <button key={tab} onClick={() => { setActiveTab(tab); if (tab === "explain" && !explainData) handleExplainRow(explainingRow); }}
                          className={`px-5 py-3 text-[13px] font-medium capitalize transition-colors ${activeTab === tab ? "border-b-2 border-brand text-brand" : "text-text-muted hover:text-text-secondary"}`}>
                          {tab === "roc" ? (trainResult.task_type === "regression" ? "Residual Plot" : "ROC Curve") : tab === "features" ? "Feature Importance" : tab}
                        </button>
                      ))}
                    </div>

                    <div className="p-5">
                      {/* Metrics tab */}
                      {activeTab === "metrics" && (
                        <div>
                          {/* Metric cards */}
                          <div className="grid grid-cols-4 md:grid-cols-5 gap-3 mb-6">
                            {trainResult.task_type === 'regression' ? (
                              <>
                                {[
                                  { k: 'r2_score', l: 'R² Score' },
                                  { k: 'mae', l: 'MAE' },
                                  { k: 'rmse', l: 'RMSE' },
                                  { k: 'mape', l: 'MAPE' },
                                ].map(({k, l}) => {
                                  const val = (currentModel.metrics as any)[k];
                                  return (
                                    <div key={k} className="bg-surface-muted rounded-xl p-3 text-center border border-border">
                                      <p className="text-[20px] font-bold text-brand">{val?.toFixed(4) || "N/A"}</p>
                                      <p className="text-[11px] text-text-muted mt-0.5">{l}</p>
                                      {currentModel.metrics.cv_mean !== null && k === "r2_score" && (
                                        <p className="text-[10px] text-text-muted mt-0.5">CV: {currentModel.metrics.cv_mean?.toFixed(4)}±{currentModel.metrics.cv_std?.toFixed(4)}</p>
                                      )}
                                    </div>
                                  );
                                })}
                              </>
                            ) : (
                              Object.entries(METRICS_LABELS).map(([key, label]) => {
                                const val = (currentModel.metrics as any)[key];
                                return (
                                  <div key={key} className="bg-surface-muted rounded-xl p-3 text-center border border-border">
                                    <p className="text-[22px] font-bold text-brand">{(val * 100).toFixed(1)}%</p>
                                    <p className="text-[11px] text-text-muted mt-0.5">{label}</p>
                                    {currentModel.metrics.cv_mean !== null && key === "accuracy" && (
                                      <p className="text-[10px] text-text-muted mt-0.5">CV: {((currentModel.metrics.cv_mean ?? 0) * 100).toFixed(1)}±{((currentModel.metrics.cv_std ?? 0) * 100).toFixed(1)}%</p>
                                    )}
                                  </div>
                                );
                              })
                            )}
                          </div>

                          {/* Model comparison chart */}
                          <h4 className="text-[13px] font-semibold text-text mb-3">Model Comparison</h4>
                          <ResponsiveContainer width="100%" height={220}>
                            <BarChart data={trainResult.task_type === 'regression' ? 
                              [
                                { metric: 'R² Score' }, { metric: 'MAE' }, { metric: 'RMSE' }, { metric: 'MAPE' }
                              ].map(r => {
                                const row: any = { metric: r.metric };
                                const key = r.metric === 'R² Score' ? 'r2_score' : r.metric.toLowerCase();
                                trainResult.models.forEach(m => {
                                  if (!m.error) row[m.name] = (m.metrics as any)[key] ?? 0;
                                });
                                return row;
                              })
                            : comparisonData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/>
                              <XAxis dataKey="metric" tick={{ fontSize: 11, fill: "#64748b" }}/>
                              {trainResult.task_type === 'regression' ? (
                                <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
                              ) : (
                                <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#64748b" }} tickFormatter={v => `${(v*100).toFixed(0)}%`}/>
                              )}
                              <Tooltip formatter={(v: any) => trainResult.task_type === 'regression' ? Number(v).toFixed(4) : `${(Number(v)*100).toFixed(1)}%`} contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}/>
                              <Legend wrapperStyle={{ fontSize: 12 }}/>
                              {trainResult.models.filter(m => !m.error).map(m => (
                                <Bar key={m.name} dataKey={m.name} fill={MODEL_COLORS[m.name]} radius={[3,3,0,0]}/>
                              ))}
                            </BarChart>
                          </ResponsiveContainer>

                          {/* Confusion Matrix */}
                          {currentModel.metrics.confusion_matrix && (
                            <div className="mt-5">
                              <h4 className="text-[13px] font-semibold text-text mb-3">Confusion Matrix</h4>
                              <div className="overflow-x-auto">
                                <table className="text-[12px] border-collapse">
                                  <thead>
                                    <tr>
                                      <th className="px-3 py-2 text-text-muted text-left">Actual \ Predicted</th>
                                      {trainResult.label_classes.map(c => (
                                        <th key={c} className="px-3 py-2 text-brand font-medium">{String(c)}</th>
                                      ))}
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {currentModel.metrics.confusion_matrix.map((row, ri) => (
                                      <tr key={ri}>
                                        <td className="px-3 py-2 font-medium text-brand">{String(trainResult.label_classes[ri])}</td>
                                        {row.map((val, ci) => (
                                          <td key={ci} className={`px-3 py-2 text-center rounded ${ri === ci ? "bg-success/10 text-success font-bold" : "bg-danger/5 text-danger"}`}>
                                            {val}
                                          </td>
                                        ))}
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* ROC / Residual tab */}
                      {activeTab === "roc" && (
                        <div>
                          {trainResult.task_type === "regression" ? (
                            <>
                              {currentModel.metrics.residual_plot && currentModel.metrics.residual_plot.length > 0 ? (
                                <>
                                  <p className="text-[12px] text-text-muted mb-3">Actual vs Predicted values (closer to diagonal line = better)</p>
                                  <ResponsiveContainer width="100%" height={320}>
                                    <LineChart data={[...currentModel.metrics.residual_plot].sort((a: any,b: any) => a.actual - b.actual)} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/>
                                      <XAxis dataKey="actual" label={{ value: "Actual", position: "bottom", fontSize: 11 }} tick={{ fontSize: 11, fill: "#64748b" }}/>
                                      <YAxis label={{ value: "Predicted", angle: -90, position: "insideLeft", fontSize: 11 }} tick={{ fontSize: 11, fill: "#64748b" }}/>
                                      <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }}/>
                                      <Line type="monotone" dataKey="actual" stroke="#cbd5e1" strokeDasharray="5 5" dot={false} strokeWidth={2} name="Perfect Prediction"/>
                                      <Line type="monotone" dataKey="predicted" stroke={MODEL_COLORS[selectedModel]} strokeWidth={0} dot={{ r: 4, fill: MODEL_COLORS[selectedModel], strokeWidth: 0 }} name="Predicted"/>
                                    </LineChart>
                                  </ResponsiveContainer>
                                </>
                              ) : (
                                <p className="text-[13px] text-text-muted">Residual plot data not available.</p>
                              )}
                            </>
                          ) : (
                            <>
                              {currentModel.metrics.roc_curve && currentModel.metrics.roc_curve.length > 0 ? (
                                <>
                                  <p className="text-[12px] text-text-muted mb-3">ROC-AUC: <span className="font-semibold text-brand">{currentModel.metrics.roc_auc?.toFixed(4)}</span> (closer to 1.0 = better)</p>
                                  <ResponsiveContainer width="100%" height={320}>
                                    <LineChart data={currentModel.metrics.roc_curve} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/>
                                      <XAxis dataKey="fpr" label={{ value: "False Positive Rate", position: "bottom", fontSize: 11 }} tick={{ fontSize: 11, fill: "#64748b" }} tickFormatter={v => v.toFixed(2)}/>
                                      <YAxis label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fontSize: 11 }} tick={{ fontSize: 11, fill: "#64748b" }} tickFormatter={v => v.toFixed(2)}/>
                                      <Tooltip formatter={(v: any) => Number(v).toFixed(3)} contentStyle={{ fontSize: 12, borderRadius: 8 }}/>
                                      <Line type="monotone" dataKey="tpr" stroke={MODEL_COLORS[selectedModel]} strokeWidth={2} dot={false} name="ROC Curve"/>
                                    </LineChart>
                                  </ResponsiveContainer>
                                </>
                              ) : (
                                <p className="text-[13px] text-text-muted">ROC curve available for binary classification only.</p>
                              )}
                            </>
                          )}
                        </div>
                      )}

                      {/* Feature Importance tab */}
                      {activeTab === "features" && (
                        <div>
                          {currentModel.feature_importance && currentModel.feature_importance.length > 0 ? (
                            <>
                              <p className="text-[12px] text-text-muted mb-3">Top features driving predictions for {selectedModel}</p>
                              <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={currentModel.feature_importance} layout="vertical" margin={{ top: 5, right: 30, left: 120, bottom: 5 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/>
                                  <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }}/>
                                  <YAxis type="category" dataKey="feature" tick={{ fontSize: 11, fill: "#64748b" }} width={110}/>
                                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }}/>
                                  <Bar dataKey="importance" fill={MODEL_COLORS[selectedModel] || "#6366f1"} radius={[0,3,3,0]}/>
                                </BarChart>
                              </ResponsiveContainer>
                            </>
                          ) : (
                            <p className="text-[13px] text-text-muted">Feature importance not available for {selectedModel}.</p>
                          )}
                        </div>
                      )}

                      {/* Predict tab */}
                      {activeTab === "predict" && (
                        <div>
                          <p className="text-[12px] text-text-muted mb-4">Enter feature values to get a real-time prediction from {selectedModel}.</p>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
                            {trainResult.feature_names.map(feat => (
                              <div key={feat}>
                                <label className="block text-[11px] text-text-muted mb-1">{feat}</label>
                                <input type="text" value={predictInputs[feat] || ""} onChange={(e) => setPredictInputs(prev => ({ ...prev, [feat]: e.target.value }))}
                                  className="w-full h-9 px-3 rounded-lg border border-border text-[13px] text-text focus:outline-none focus:border-brand/40"
                                  placeholder="0"/>
                              </div>
                            ))}
                          </div>
                          
                          <div className="flex gap-2">
                              <button onClick={handlePredict} disabled={predicting}
                                className="px-5 h-9 rounded-lg bg-brand hover:bg-brand-light text-white text-[13px] font-medium disabled:opacity-40 transition-colors">
                                {predicting ? "Predicting..." : "Predict"}
                              </button>
                              
                              <button onClick={() => batchFileRef.current?.click()} disabled={batchPredicting}
                                className="px-5 h-9 rounded-lg border border-border text-text hover:bg-surface-muted text-[13px] font-medium disabled:opacity-40 transition-colors">
                                {batchPredicting ? "Running Batch..." : "Batch Predict (CSV)"}
                              </button>
                              <input ref={batchFileRef} type="file" accept=".csv" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleBatchPredict(f); }}/>
                          </div>
                          
                          {batchMsg && <p className="text-[12px] text-brand mt-3">{batchMsg}</p>}

                          {prediction && (
                            <div className="mt-5 p-4 rounded-xl bg-surface-muted border border-border">
                              <div className="flex items-center justify-between mb-3">
                                <p className="text-[12px] text-text-muted">Prediction Result</p>
                                <button onClick={() => handleExplainPrediction(prediction.featuresUsed)} className="text-[11px] px-2 py-1 bg-white border border-border rounded text-brand hover:border-brand/30">
                                  Why this prediction?
                                </button>
                              </div>
                              <div className="flex items-center gap-3 mb-4">
                                <div className="px-4 py-2 rounded-xl bg-brand text-white font-bold text-[18px]">
                                  {trainResult.task_type === 'regression' ? Number(prediction.prediction).toFixed(4) : prediction.prediction}
                                </div>
                                <div>
                                  {trainResult.task_type !== 'regression' && (
                                    <p className="text-[13px] font-semibold text-text">{prediction.confidence}% confident</p>
                                  )}
                                  <p className="text-[11px] text-text-muted">Model: {prediction.model_used}</p>
                                </div>
                              </div>
                              {trainResult.task_type !== 'regression' && prediction.probabilities && (
                                <div className="space-y-2">
                                  {Object.entries(prediction.probabilities).map(([cls, prob]: [string, any]) => (
                                    <div key={cls} className="flex items-center gap-2">
                                      <span className="text-[12px] text-text-secondary w-20 truncate">{cls}</span>
                                      <div className="flex-1 h-2 bg-border rounded-full overflow-hidden">
                                        <div className="h-full bg-brand rounded-full transition-all" style={{ width: `${prob}%` }}/>
                                      </div>
                                      <span className="text-[12px] text-text-secondary w-12 text-right">{prob}%</span>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Explain tab */}
                      {activeTab === "explain" && (
                          <div>
                              <div className="flex justify-between items-center mb-4">
                                  <p className="text-[12px] text-text-muted">SHAP value contributions to the final prediction.</p>
                                  <div className="flex items-center gap-2 bg-surface-muted border border-border rounded px-2 py-1">
                                      <button onClick={() => handleExplainRow(Math.max(0, explainingRow - 1))} disabled={explaining} className="text-[12px] px-1 hover:text-brand disabled:opacity-50">◀</button>
                                      <span className="text-[12px] text-text-secondary">Row #{explainingRow}</span>
                                      <button onClick={() => handleExplainRow(explainingRow + 1)} disabled={explaining} className="text-[12px] px-1 hover:text-brand disabled:opacity-50">▶</button>
                                  </div>
                              </div>
                              
                              {explaining && <div className="py-10 text-center text-[13px] text-text-muted"><div className="w-6 h-6 border-2 border-brand/30 border-t-brand rounded-full animate-spin mx-auto mb-2"/> Computing SHAP values...</div>}
                              
                              {!explaining && explainData && (
                                  <>
                                      <div className="mb-4 bg-white border border-border rounded-lg p-3 flex justify-between items-center">
                                          <div>
                                              <p className="text-[11px] text-text-muted">Base Value</p>
                                              <p className="text-[14px] font-medium text-text">{explainData.base_value?.toFixed(4)}</p>
                                          </div>
                                          <div className="text-right">
                                              <p className="text-[11px] text-text-muted">Final Prediction</p>
                                              <p className="text-[16px] font-bold text-brand">
                                                  {trainResult.task_type === "regression" ? explainData.predicted_value?.toFixed(4) : explainData.prediction_label}
                                              </p>
                                          </div>
                                      </div>
                                      
                                      <ResponsiveContainer width="100%" height={350}>
                                        <BarChart data={explainData.feature_contributions} layout="vertical" margin={{ top: 5, right: 30, left: 120, bottom: 5 }}>
                                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/>
                                          <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }}/>
                                          <YAxis type="category" dataKey="feature" tickFormatter={(v, i) => `${v} = ${explainData.feature_contributions[i]?.feature_value}`} tick={{ fontSize: 11, fill: "#64748b" }} width={110}/>
                                          <Tooltip formatter={(v: any) => Number(v).toFixed(4)} contentStyle={{ fontSize: 12, borderRadius: 8 }}/>
                                          <Bar dataKey="shap_value" radius={2}>
                                            {explainData.feature_contributions.map((entry: any, index: number) => (
                                              <Cell key={`cell-${index}`} fill={entry.shap_value >= 0 ? "#6366f1" : "#ef4444"} />
                                            ))}
                                          </Bar>
                                        </BarChart>
                                      </ResponsiveContainer>
                                      <div className="flex justify-center gap-4 mt-2">
                                          <span className="flex items-center gap-1 text-[11px] text-text-muted"><div className="w-3 h-3 bg-indigo-500 rounded-sm"></div> Pushes Prediction Higher</span>
                                          <span className="flex items-center gap-1 text-[11px] text-text-muted"><div className="w-3 h-3 bg-red-500 rounded-sm"></div> Pushes Prediction Lower</span>
                                      </div>
                                  </>
                              )}
                          </div>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>

      <footer className="border-t border-border py-5 mt-12">
        <p className="text-center text-[12px] text-text-muted">
          PredictFlow · AutoML with Logistic Regression, Random Forest, XGBoost & Neural Networks · Built by Badal Gupta
        </p>
      </footer>
    </div>
  );
}

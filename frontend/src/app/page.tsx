"use client";
import { useState, useRef, useCallback } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend
} from "recharts";
import { uploadCSV, trainModels, predict, deleteDataset } from "@/lib/api";

/* ---- Types ---- */
interface ColInfo { name: string; type: string; null_count: number; unique_count: number; sample_values: any[]; }
interface Dataset { filename: string; row_count: number; column_count: number; columns: ColInfo[]; }
interface Metrics { accuracy: number; precision: number; recall: number; f1_score: number; roc_auc: number; cv_mean: number | null; cv_std: number | null; confusion_matrix: number[][]; roc_curve: { fpr: number; tpr: number }[]; }
interface ModelResult { name: string; metrics: Metrics; feature_importance: { feature: string; importance: number }[]; is_best: boolean; error?: string; }
interface TrainResult { models: ModelResult[]; best_model: string; target_col: string; label_classes: string[]; num_classes: number; feature_names: string[]; training_samples: number; test_samples: number; }

const MODEL_COLORS: Record<string, string> = {
  "Logistic Regression": "#6366f1",
  "Random Forest": "#059669",
  "XGBoost": "#d97706",
  "Neural Network": "#dc2626",
};

const METRICS_LABELS: Record<string, string> = {
  accuracy: "Accuracy", precision: "Precision", recall: "Recall",
  f1_score: "F1 Score", roc_auc: "ROC-AUC",
};

export default function Home() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [target, setTarget] = useState("");
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"metrics" | "roc" | "features" | "predict">("metrics");
  const [uploading, setUploading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [predictInputs, setPredictInputs] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  /* Upload */
  const handleUpload = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".csv")) { setError("Only CSV files accepted."); return; }
    setUploading(true); setError(null); setTrainResult(null); setTarget("");
    try {
      const res = await uploadCSV(file);
      setDataset(res);
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
    setTraining(true); setError(null); setTrainResult(null); setPrediction(null);
    try {
      const res = await trainModels(target);
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
    setPredicting(true); setError(null);
    try {
      const features: Record<string, number> = {};
      Object.entries(predictInputs).forEach(([k, v]) => { features[k] = parseFloat(v) || 0; });
      const res = await predict(features, selectedModel);
      setPrediction(res);
    } catch (e: any) { setError(e.message); }
    finally { setPredicting(false); }
  };

  /* Clear */
  const handleClear = async () => {
    await deleteDataset().catch(() => {});
    setDataset(null); setTrainResult(null); setTarget(""); setError(null); setPrediction(null);
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

            {/* Target selector + Train */}
            {!trainResult && (
              <div className="bg-white rounded-xl border border-border p-5">
                <h3 className="text-[14px] font-semibold text-text mb-1">Select Target Column</h3>
                <p className="text-[12px] text-text-muted mb-4">Choose the column you want to predict. Must be a classification target (2–20 unique values).</p>
                <div className="flex gap-3">
                  <select value={target} onChange={(e) => setTarget(e.target.value)}
                    className="flex-1 h-10 px-3 rounded-lg border border-border text-[13px] text-text focus:outline-none focus:border-brand/40">
                    <option value="">-- Select target column --</option>
                    {dataset.columns.filter(c => c.unique_count >= 2 && c.unique_count <= 20).map(c => (
                      <option key={c.name} value={c.name}>{c.name} ({c.unique_count} classes)</option>
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
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: MODEL_COLORS[trainResult.best_model] + "20" }}>
                    <span className="text-lg">🏆</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-[13px] font-semibold text-text">Best Model: {trainResult.best_model}</p>
                    <p className="text-[12px] text-text-muted">
                      ROC-AUC: {trainResult.models.find(m => m.name === trainResult.best_model)?.metrics.roc_auc.toFixed(4)} · 
                      Accuracy: {trainResult.models.find(m => m.name === trainResult.best_model)?.metrics.accuracy.toFixed(4)} · 
                      {trainResult.training_samples} training / {trainResult.test_samples} test samples
                    </p>
                  </div>
                  <button onClick={() => { setTrainResult(null); setPrediction(null); }}
                    className="text-[12px] px-3 h-8 rounded-lg border border-border text-text-muted hover:text-brand hover:border-brand/30 transition-colors">
                    Retrain
                  </button>
                </div>

                {/* Model selector */}
                <div className="flex gap-2 flex-wrap">
                  {trainResult.models.filter(m => !m.error).map(m => (
                    <button key={m.name} onClick={() => setSelectedModel(m.name)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-xl border text-[12px] font-medium transition-all ${selectedModel === m.name ? "border-brand bg-brand/[0.06] text-brand" : "border-border text-text-secondary hover:border-brand/30"}`}>
                      <span className="w-2.5 h-2.5 rounded-full" style={{ background: MODEL_COLORS[m.name] }}/>
                      {m.name}
                      {m.is_best && <span className="text-[10px] bg-brand/10 text-brand px-1.5 py-0.5 rounded">Best</span>}
                    </button>
                  ))}
                </div>

                {currentModel && !currentModel.error && (
                  <div className="bg-white rounded-xl border border-border overflow-hidden">
                    {/* Tabs */}
                    <div className="flex border-b border-border">
                      {(["metrics", "roc", "features", "predict"] as const).map(tab => (
                        <button key={tab} onClick={() => setActiveTab(tab)}
                          className={`px-5 py-3 text-[13px] font-medium capitalize transition-colors ${activeTab === tab ? "border-b-2 border-brand text-brand" : "text-text-muted hover:text-text-secondary"}`}>
                          {tab === "roc" ? "ROC Curve" : tab === "features" ? "Feature Importance" : tab === "predict" ? "Predict" : "Metrics"}
                        </button>
                      ))}
                    </div>

                    <div className="p-5">
                      {/* Metrics tab */}
                      {activeTab === "metrics" && (
                        <div>
                          {/* Metric cards */}
                          <div className="grid grid-cols-5 gap-3 mb-6">
                            {Object.entries(METRICS_LABELS).map(([key, label]) => {
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
                            })}
                          </div>

                          {/* Model comparison chart */}
                          <h4 className="text-[13px] font-semibold text-text mb-3">Model Comparison</h4>
                          <ResponsiveContainer width="100%" height={220}>
                            <BarChart data={comparisonData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/>
                              <XAxis dataKey="metric" tick={{ fontSize: 11, fill: "#64748b" }}/>
                              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#64748b" }} tickFormatter={v => `${(v*100).toFixed(0)}%`}/>
                              <Tooltip formatter={(v: any) => `${(v*100).toFixed(1)}%`} contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}/>
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

                      {/* ROC tab */}
                      {activeTab === "roc" && (
                        <div>
                          {currentModel.metrics.roc_curve && currentModel.metrics.roc_curve.length > 0 ? (
                            <>
                              <p className="text-[12px] text-text-muted mb-3">ROC-AUC: <span className="font-semibold text-brand">{currentModel.metrics.roc_auc.toFixed(4)}</span> (closer to 1.0 = better)</p>
                              <ResponsiveContainer width="100%" height={320}>
                                <LineChart data={currentModel.metrics.roc_curve} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/>
                                  <XAxis dataKey="fpr" label={{ value: "False Positive Rate", position: "bottom", fontSize: 11 }} tick={{ fontSize: 11, fill: "#64748b" }} tickFormatter={v => v.toFixed(2)}/>
                                  <YAxis label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fontSize: 11 }} tick={{ fontSize: 11, fill: "#64748b" }} tickFormatter={v => v.toFixed(2)}/>
                                  <Tooltip formatter={(v: any) => v.toFixed(3)} contentStyle={{ fontSize: 12, borderRadius: 8 }}/>
                                  <Line type="monotone" dataKey="tpr" stroke={MODEL_COLORS[selectedModel]} strokeWidth={2} dot={false} name="ROC Curve"/>
                                </LineChart>
                              </ResponsiveContainer>
                            </>
                          ) : (
                            <p className="text-[13px] text-text-muted">ROC curve available for binary classification only.</p>
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
                                  <Bar dataKey="importance" fill={MODEL_COLORS[selectedModel]} radius={[0,3,3,0]}/>
                                </BarChart>
                              </ResponsiveContainer>
                            </>
                          ) : (
                            <p className="text-[13px] text-text-muted">Feature importance not available for Neural Network.</p>
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
                                <input type="number" value={predictInputs[feat] || ""} onChange={(e) => setPredictInputs(prev => ({ ...prev, [feat]: e.target.value }))}
                                  className="w-full h-9 px-3 rounded-lg border border-border text-[13px] text-text focus:outline-none focus:border-brand/40"
                                  placeholder="0"/>
                              </div>
                            ))}
                          </div>
                          <button onClick={handlePredict} disabled={predicting}
                            className="px-5 h-9 rounded-lg bg-brand hover:bg-brand-light text-white text-[13px] font-medium disabled:opacity-40 transition-colors">
                            {predicting ? "Predicting..." : "Predict"}
                          </button>

                          {prediction && (
                            <div className="mt-5 p-4 rounded-xl bg-surface-muted border border-border">
                              <p className="text-[12px] text-text-muted mb-3">Prediction Result</p>
                              <div className="flex items-center gap-3 mb-4">
                                <div className="px-4 py-2 rounded-xl bg-brand text-white font-bold text-[18px]">{prediction.prediction}</div>
                                <div>
                                  <p className="text-[13px] font-semibold text-text">{prediction.confidence}% confident</p>
                                  <p className="text-[11px] text-text-muted">Model: {prediction.model_used}</p>
                                </div>
                              </div>
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
                            </div>
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

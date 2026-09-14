const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:10000";

function getSessionId(): string {
  if (typeof window === "undefined") return "default";
  let id = localStorage.getItem("predictflow_session");
  if (!id) { id = crypto.randomUUID(); localStorage.setItem("predictflow_session", id); }
  return id;
}

async function request(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "X-Session-ID": getSessionId(), ...(options?.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const uploadCSV = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return request("/api/upload", { method: "POST", body: form });
};

export const trainModels = (target_column: string, feature_columns?: string[]) =>
  request("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_column, feature_columns }),
  });

export const predict = (features: Record<string, number>, model_name = "best") =>
  request("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features, model_name }),
  });

export const deleteDataset = () => request("/api/dataset", { method: "DELETE" });

export const batchPredict = async (file: File, model_name = "best") => {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", model_name);
  const res = await fetch(`${API_BASE}/api/predict/batch`, {
    method: "POST",
    headers: { "X-Session-ID": getSessionId() },
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Batch prediction failed: ${res.status}`);
  }
  return res.blob();
};

export const profileDataset = () => request("/api/profile", { method: "POST" });

export const explainPrediction = (model_name = "best", row_index = -1, features?: Record<string, number>) =>
  request("/api/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name, row_index, features }),
  });

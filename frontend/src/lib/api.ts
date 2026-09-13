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

export const trainModels = (target_column: string) =>
  request("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_column }),
  });

export const predict = (features: Record<string, number>, model_name = "best") =>
  request("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features, model_name }),
  });

export const deleteDataset = () => request("/api/dataset", { method: "DELETE" });

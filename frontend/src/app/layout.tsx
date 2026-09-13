import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "PredictFlow — AutoML Model Training & Evaluation",
  description: "Upload a CSV, select a target column, and PredictFlow automatically trains, evaluates, and compares multiple ML models.",
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}

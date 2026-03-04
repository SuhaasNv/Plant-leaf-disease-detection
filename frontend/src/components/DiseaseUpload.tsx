"use client";

import { useState, useRef } from "react";

// API returns {"class": string, "confidence": float}
type PredictionResult = {
  class: string;
  confidence: number;
};

type Props = { apiUrl: string };

function formatLabel(raw: string) {
  return raw
    .replace(/___/g, " — ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function DiseaseUpload({ apiUrl }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (selected: File | undefined) => {
    setResult(null);
    setError(null);
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please select an image file (PNG, JPG, JPEG).");
      return;
    }
    setFile(selected);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(selected);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60_000);

    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body,
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const detail = err.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail[0]?.msg || res.statusText
              : res.statusText || "Prediction failed."
        );
      }

      setResult(await res.json());
    } catch (e) {
      clearTimeout(timer);
      if (!(e instanceof Error)) { setError("Something went wrong."); return; }
      if (e.name === "AbortError")
        setError("Request timed out — the model may still be loading. Try again.");
      else if (e.message.toLowerCase().includes("fetch") || e.message.toLowerCase().includes("network"))
        setError("Cannot reach the API. Make sure the backend is running on port 8000.");
      else
        setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const confidencePct = result ? Math.round(result.confidence * 100) : 0;
  const isHealthy = result?.["class"]?.toLowerCase().includes("healthy");

  return (
    <div className="space-y-4">

      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        className={`flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
          dragging
            ? "border-green-400 bg-green-50"
            : "border-gray-200 bg-gray-50 hover:border-green-300 hover:bg-green-50/40"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          onChange={(e) => handleFile(e.target.files?.[0])}
          className="hidden"
        />

        {preview ? (
          <div className="space-y-3">
            <img
              src={preview}
              alt="Leaf preview"
              className="mx-auto max-h-52 rounded-xl object-contain shadow-sm"
            />
            <p className="text-xs text-gray-400">{file?.name}</p>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-green-50 text-2xl">
              🍃
            </div>
            <p className="font-medium text-gray-700">
              {dragging ? "Drop it here" : "Drag & drop or click to upload"}
            </p>
            <p className="text-sm text-gray-400">PNG, JPG or JPEG</p>
          </div>
        )}
      </div>

      {/* Actions */}
      {file && (
        <div className="flex gap-3">
          <button
            onClick={handlePredict}
            disabled={loading}
            className="flex-1 rounded-lg bg-green-600 py-2.5 text-sm font-semibold text-white
              shadow-sm transition-[transform,box-shadow,background-color] duration-150
              hover:-translate-y-0.5 hover:bg-green-700 hover:shadow-md
              active:translate-y-0 active:scale-[0.98] active:shadow-sm
              disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0
              sm:flex-none sm:px-8"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                {/* Arc spinner — cleaner than the filled-path version */}
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <circle cx="12" cy="12" r="9" className="opacity-20" />
                  <path d="M12 3a9 9 0 0 1 9 9" />
                </svg>
                Analyzing…
              </span>
            ) : (
              "Analyze"
            )}
          </button>
          <button
            onClick={handleClear}
            className="rounded-lg border border-gray-200 bg-white px-5 py-2.5 text-sm font-medium text-gray-600
              transition-[transform,background-color] duration-150
              hover:-translate-y-0.5 hover:bg-gray-50
              active:translate-y-0 active:scale-[0.98]"
          >
            Clear
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-red-100 bg-red-50 p-4">
          <span className="mt-0.5 text-red-500">⚠</span>
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Result — fades + slides up when it appears */}
      {result && (
        <div className="animate-fade-in-up rounded-2xl bg-white p-5 shadow-xl sm:p-8">
          <p className="text-xs font-medium uppercase tracking-widest text-gray-400">
            Diagnosis
          </p>

          {/* Class name + status badge */}
          <div className="mt-3 flex flex-wrap items-start justify-between gap-3">
            <h3 className="text-xl font-bold leading-tight text-gray-900 sm:text-3xl">
              {formatLabel(result["class"])}
            </h3>
            <span
              className={`shrink-0 rounded-full px-3 py-1 text-sm font-medium ${
                isHealthy ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-700"
              }`}
            >
              {isHealthy ? "Healthy ✓" : "Disease Detected"}
            </span>
          </div>

          {/* Confidence bar */}
          <div className="mt-5">
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-gray-400">Confidence</span>
              <span
                className={`text-2xl font-bold tabular-nums ${
                  confidencePct >= 80
                    ? "text-green-600"
                    : confidencePct >= 60
                      ? "text-yellow-500"
                      : "text-red-500"
                }`}
              >
                {confidencePct}%
              </span>
            </div>

            <div className="mt-2 h-3 w-full overflow-hidden rounded-full bg-gray-100">
              <div
                className={`h-full rounded-full transition-[width] duration-700 ease-out ${
                  confidencePct >= 80
                    ? "bg-green-500"
                    : confidencePct >= 60
                      ? "bg-yellow-400"
                      : "bg-red-400"
                }`}
                style={{ width: `${confidencePct}%` }}
              />
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

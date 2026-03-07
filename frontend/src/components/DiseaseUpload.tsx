"use client";

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

type PredictionResult = {
  class?: string;
  cls?: string;
  confidence: number;
};

function formatLabel(raw: string | undefined | null): string {
  if (raw == null || typeof raw !== "string") return "Unknown";
  return raw
    .replace(/___/g, " — ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// Labels shown at different points in the progress bar
const LOADING_LABELS = [
  { at: 0, text: "Scanning leaf…" },
  { at: 35, text: "Analysing patterns…" },
  { at: 70, text: "Identifying disease…" },
  { at: 92, text: "Almost done…" },
];

function loadingLabel(pct: number) {
  let label = LOADING_LABELS[0].text;
  for (const { at, text } of LOADING_LABELS) {
    if (pct >= at) label = text;
  }
  return label;
}

// ── progress animation ──────────────────────────────────────────────────────
// Runs for ANIM_MS ms (0 → 100), applying an ease that rushes early then slows.
const ANIM_MS = 3400;
const TICK_MS = 30;

function easeProgress(t: number) {
  // Fast start, slow finish
  return 1 - Math.pow(1 - t, 2.4);
}

export function DiseaseUpload({ onDisease }: { onDisease?: (disease: string | null) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Mirror of `file` state in a ref — guarantees handlePredict always closes
  // over the *current* file even if React batches the state update.
  const fileRef = useRef<File | null>(null);

  // Refs to coordinate the parallel animation + API fetch
  const animDoneRef = useRef(false);
  const pendingRef = useRef<PredictionResult | null>(null);
  const pendingErrRef = useRef<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleFile = (selected: File | undefined) => {
    setResult(null);
    setError(null);
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please select an image file (PNG, JPG, JPEG).");
      return;
    }

    // ── FIX: reset the input value immediately so that picking the same
    // filename a second time still fires onChange (browser suppresses it
    // otherwise because the value hasn't "changed").
    if (inputRef.current) inputRef.current.value = "";

    console.log(`[upload] file selected: name=${selected.name} size=${selected.size} type=${selected.type}`);

    fileRef.current = selected;   // keep ref in sync before React re-render
    setFile(selected);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(selected);
  };

  const revealResult = (res: PredictionResult | null, err: string | null) => {
    setProgress(100);
    setTimeout(() => {
      setLoading(false);
      setProgress(0);
      if (err) {
        setError(err);
      } else {
        setResult(res);
        onDisease?.(res?.["class"] ?? res?.cls ?? null);
      }
    }, 350); // brief pause at 100% before showing result
  };

  const handlePredict = async () => {
    // ── FIX: read from ref, not state, to guarantee the latest File object
    // (avoids stale closure if React batched the setFile update).
    const currentFile = fileRef.current;
    if (!currentFile) return;

    console.log(`[upload] predict triggered: file=${currentFile.name} size=${currentFile.size}`);

    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);
    animDoneRef.current = false;
    pendingRef.current = null;
    pendingErrRef.current = null;

    // ── 1. Start progress animation ──────────────────────────────────────
    const startTime = Date.now();
    intervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const t = Math.min(elapsed / ANIM_MS, 1);
      const pct = Math.round(easeProgress(t) * 100);
      setProgress(pct);

      if (t >= 1) {
        clearInterval(intervalRef.current!);
        animDoneRef.current = true;

        // If API already returned, reveal now
        if (pendingRef.current !== null || pendingErrRef.current !== null) {
          revealResult(pendingRef.current, pendingErrRef.current);
        }
      }
    }, TICK_MS);

    // ── 2. Fire API call in parallel ─────────────────────────────────────
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60_000);

    try {
      const body = new FormData();
      body.append("file", currentFile);

      console.log(`[upload] FormData ready: appended file=${currentFile.name} size=${currentFile.size}`);

      const res = await fetch("/api/predict", {
        method: "POST",
        body,
        cache: "no-store",
        signal: controller.signal,
      });

      console.log(`[upload] fetch response: status=${res.status}`);
      clearTimeout(timeout);

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

      const data: PredictionResult = await res.json();
      const label = data["class"] ?? data.cls;
      console.log(`[upload] prediction received: class=${label} confidence=${data.confidence}`);

      if (animDoneRef.current) {
        revealResult(data, null);
      } else {
        pendingRef.current = data; // wait for animation to finish
      }
    } catch (e) {
      clearTimeout(timeout);
      clearInterval(intervalRef.current!);

      let msg = "Something went wrong.";
      if (e instanceof Error) {
        if (e.name === "AbortError")
          msg = "Request timed out — the model may still be loading. Try again.";
        else if (e.message.toLowerCase().includes("fetch") || e.message.toLowerCase().includes("network"))
          msg = "Cannot reach the prediction service. Please try again later.";
        else
          msg = e.message;
      }

      if (animDoneRef.current) {
        revealResult(null, msg);
      } else {
        pendingErrRef.current = msg;
      }
    }
  };

  const handleClear = () => {
    clearInterval(intervalRef.current!);
    fileRef.current = null;
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setProgress(0);
    onDisease?.(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const confidencePct = result ? Math.round(result.confidence * 100) : 0;
  const isHealthy = (result?.["class"] ?? result?.cls)?.toLowerCase().includes("healthy");

  return (
    <div className="space-y-4">

      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        onClick={() => !loading && inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && !loading && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        className={`flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${dragging
          ? "border-green-400 bg-green-50"
          : "border-gray-200 bg-gray-50 hover:border-green-300 hover:bg-green-50/40"
          } ${loading ? "pointer-events-none opacity-60" : ""}`}
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
      {file && !loading && (
        <div className="flex gap-3">
          <button
            onClick={handlePredict}
            className="flex-1 rounded-lg bg-green-600 py-2.5 text-sm font-semibold text-white
              shadow-sm transition-[transform,box-shadow,background-color] duration-150
              hover:-translate-y-0.5 hover:bg-green-700 hover:shadow-md
              active:translate-y-0 active:scale-[0.98] active:shadow-sm
              sm:flex-none sm:px-8"
          >
            Analyze
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

      {/* Progress bar — shown while loading */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
            className="rounded-2xl border border-gray-100 bg-white p-5 shadow-lg sm:p-6"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-gray-700">
                {loadingLabel(progress)}
              </span>
              <span className="text-sm font-bold tabular-nums text-green-600">
                {progress}%
              </span>
            </div>

            {/* Track */}
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-gray-100">
              {/* Fill */}
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-green-500 to-green-400"
                animate={{ width: `${progress}%` }}
                transition={{ duration: TICK_MS / 1000, ease: "linear" }}
              />
            </div>

            {/* Animated leaf dots */}
            <div className="mt-3 flex items-center gap-1.5">
              {[0, 1, 2].map((i) => (
                <motion.span
                  key={i}
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.4 }}
                  className="text-green-500 text-xs"
                >
                  🍃
                </motion.span>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex items-start gap-3 rounded-xl border border-red-100 bg-red-50 p-4"
          >
            <span className="mt-0.5 text-red-500">⚠</span>
            <p className="text-sm text-red-700">{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="rounded-2xl bg-white p-5 shadow-xl sm:p-8"
          >
            <p className="text-xs font-medium uppercase tracking-widest text-gray-400">
              Diagnosis
            </p>

            <div className="mt-3 flex flex-wrap items-start justify-between gap-3">
              <h3 className="text-xl font-bold leading-tight text-gray-900 sm:text-3xl">
                {formatLabel(result["class"] ?? result.cls)}
              </h3>
              <span
                className={`shrink-0 rounded-full px-3 py-1 text-sm font-medium ${isHealthy ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-700"
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
                  className={`text-2xl font-bold tabular-nums ${confidencePct >= 80
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
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${confidencePct}%` }}
                  transition={{ duration: 0.9, ease: "easeOut", delay: 0.2 }}
                  className={`h-full rounded-full ${confidencePct >= 80
                    ? "bg-green-500"
                    : confidencePct >= 60
                      ? "bg-yellow-400"
                      : "bg-red-400"
                    }`}
                />
              </div>
            </div>

            {/* Nudge to use Assistant */}
            {!isHealthy && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2, duration: 0.4 }}
                className="mt-6 flex items-center gap-3 rounded-lg bg-green-50 p-4 border border-green-100"
              >
                <span className="text-xl">💡</span>
                <p className="text-sm text-green-800">
                  Want to know how to treat this? Click the <strong>Plant Assistant</strong> icon in the bottom right for advice!
                </p>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}

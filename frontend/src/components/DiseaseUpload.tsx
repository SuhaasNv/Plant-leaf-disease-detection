"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export type PredictionItem = {
  label: string;
  confidence: number;
  treatment?: string[];
  prevention?: string[];
};

export type PredictionResult = {
  predictions: PredictionItem[];
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

export function DiseaseUpload({ onDisease }: { onDisease?: (predictions: PredictionItem[] | null) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Cleanup camera stream on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      streamRef.current = stream;
      setIsCameraOpen(true);
      // We set srcObject inside a small timeout or wait for React to render the <video>
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      }, 50);
    } catch (e: any) {
      setError("Unable to access camera. Please allow camera permissions.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setIsCameraOpen(false);
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setError("Failed to capture image.");
        stopCamera();
        return;
      }
      const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
      stopCamera();
      handleFile(file);
    }, "image/jpeg", 0.9);
  };

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
        onDisease?.(res?.predictions ?? null);
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
        const detail = err.detail || err.error;
        throw new Error(
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail[0]?.msg || res.statusText
              : res.statusText || "Prediction failed."
        );
      }

      const data: PredictionResult = await res.json();
      const label = data.predictions?.[0]?.label ?? "Unknown";
      console.log(`[upload] prediction received: class=${label} confidence=${data.predictions?.[0]?.confidence}`);

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
    stopCamera();
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

  const topPrediction = result?.predictions?.[0];
  const confidencePct = topPrediction ? Math.round(topPrediction.confidence * 100) : 0;
  const isHealthy = topPrediction?.label?.toLowerCase().includes("healthy");

  return (
    <div className="space-y-4">

      {/* Drop zone / Camera View */}
      {isCameraOpen ? (
        <div className="relative flex min-h-56 flex-col items-center justify-center overflow-hidden rounded-2xl bg-black shadow-inner">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="h-64 w-full object-cover sm:h-80"
          />
          <div className="absolute bottom-4 flex gap-3">
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); capturePhoto(); }}
              className="rounded-full bg-green-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg hover:bg-green-700 active:scale-95 transition-transform"
            >
              Take Photo
            </button>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); stopCamera(); }}
              className="rounded-full bg-white/20 backdrop-blur-md px-6 py-2.5 text-sm font-semibold text-white shadow-lg hover:bg-white/30 active:scale-95 transition-transform border border-white/30"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
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
                className="mx-auto max-h-48 sm:max-h-52 w-full sm:w-auto rounded-xl object-contain shadow-sm"
              />
              <p className="text-xs text-gray-400">{file?.name}</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-green-50 text-2xl">
                🍃
              </div>
              <div className="space-y-1">
                <p className="font-medium text-gray-700">
                  {dragging ? "Drop it here" : "Drag & drop or click to upload"}
                </p>
                <p className="text-sm text-gray-400">PNG, JPG or JPEG</p>
              </div>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  startCamera();
                }}
                className="inline-flex mt-2 items-center gap-2 rounded-lg bg-green-100 px-4 py-2 text-sm font-medium text-green-700 transition-[transform,colors] hover:bg-green-200 active:scale-95"
              >
                <span>📷</span> Open Camera
              </button>
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      {file && !loading && (
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            onClick={handlePredict}
            className="flex-1 rounded-xl bg-green-600 py-3.5 sm:py-2.5 text-base sm:text-sm font-semibold text-white
              shadow-sm transition-[transform,box-shadow,background-color] duration-150
              hover:-translate-y-0.5 hover:bg-green-700 hover:shadow-md
              active:translate-y-0 active:scale-[0.98] active:shadow-sm
              sm:flex-none sm:px-8"
          >
            Analyze
          </button>
          <button
            onClick={handleClear}
            className="rounded-xl border border-gray-200 bg-white px-5 py-3.5 sm:py-2.5 text-base sm:text-sm font-medium text-gray-600
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
                {formatLabel(topPrediction?.label)}
              </h3>
              <span
                className={`shrink-0 rounded-full px-3 py-1 text-sm font-medium ${isHealthy ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-700"
                  }`}
              >
                {isHealthy ? "Healthy ✓" : "Disease Detected"}
              </span>
            </div>

            {/* Predictions List */}
            <div className="mt-6 lg:mt-8 space-y-3">
              {result.predictions?.slice().sort((a, b) => b.confidence - a.confidence).map((pred, idx) => {
                const predPct = Math.round(pred.confidence * 100);
                const isTop = idx === 0;

                return (
                  <div
                    key={idx}
                    className={`rounded-xl p-4 transition-all border ${isTop
                      ? 'border-green-100 bg-green-50/50 shadow-sm'
                      : 'border-gray-100 bg-gray-50'
                      }`}
                  >
                    <div className="flex items-center justify-between gap-4 mb-3">
                      <span className={`text-sm sm:text-base font-medium leading-tight ${isTop ? 'text-gray-900' : 'text-gray-600'}`}>
                        {formatLabel(pred.label)}
                      </span>
                      <span
                        className={`text-sm sm:text-base font-bold tabular-nums shrink-0 ${isTop
                          ? predPct >= 80 ? 'text-green-600' : predPct >= 60 ? 'text-yellow-600' : 'text-red-500'
                          : 'text-gray-500'
                          }`}
                      >
                        {predPct}%
                      </span>
                    </div>

                    <div className={`w-full overflow-hidden rounded-full ${isTop ? 'h-2 bg-gray-200/60' : 'h-1.5 bg-gray-200'}`}>
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${predPct}%` }}
                        transition={{ duration: 0.9, ease: "easeOut", delay: 0.2 + idx * 0.15 }}
                        className={`h-full rounded-full ${isTop
                          ? predPct >= 80 ? 'bg-green-500' : predPct >= 60 ? 'bg-yellow-400' : 'bg-red-400'
                          : 'bg-gray-300'
                          }`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Treatment & Prevention Card */}
            {(result.predictions?.[0]?.treatment?.length || result.predictions?.[0]?.prevention?.length) ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.0, duration: 0.4 }}
                className="mt-6 rounded-xl border border-blue-100 bg-blue-50 p-5 shadow-sm"
              >
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-xl">🩺</span>
                  <h3 className="font-bold text-blue-900">Treatment & Prevention</h3>
                </div>
                <div className="space-y-4">
                  {result.predictions[0].treatment && result.predictions[0].treatment.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-blue-800 mb-2">How to Treat:</h4>
                      <ul className="list-disc pl-5 space-y-1 text-sm text-blue-800/80">
                        {result.predictions[0].treatment.map((it, i) => <li key={i}>{it}</li>)}
                      </ul>
                    </div>
                  )}
                  {result.predictions[0].prevention && result.predictions[0].prevention.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-blue-800 mb-2">How to Prevent:</h4>
                      <ul className="list-disc pl-5 space-y-1 text-sm text-blue-800/80">
                        {result.predictions[0].prevention.map((it, i) => <li key={i}>{it}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              </motion.div>
            ) : null}

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

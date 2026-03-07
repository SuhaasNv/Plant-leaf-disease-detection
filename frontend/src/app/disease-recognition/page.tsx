"use client";

import { useState } from "react";
import { DiseaseUpload } from "@/components/DiseaseUpload";
import { PlantAssistant } from "@/components/PlantAssistant";

export default function DiseaseRecognitionPage() {
  const [detectedDisease, setDetectedDisease] = useState<string | null>(null);

  return (
    <>
      <div className="mx-auto max-w-2xl px-4 py-10 sm:px-6 sm:py-16">

        {/* Header */}
        <div className="text-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-green-200 bg-green-50 px-4 py-1.5 text-xs font-medium text-green-600 sm:text-sm">
            <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
            AI Disease Detection
          </span>
          <h1 className="mt-4 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Analyze Your Plant
          </h1>
          <p className="mt-3 text-sm text-gray-600 sm:text-base">
            Upload a clear photo of a leaf. Our model will identify any disease in under a second.
          </p>
        </div>

        {/* Upload card */}
        <div className="mt-8 rounded-2xl bg-white p-4 shadow-xl sm:mt-10 sm:p-8">
          <DiseaseUpload onDisease={setDetectedDisease} />
        </div>

        {/* Footer hint */}
        <p className="mt-5 text-center text-xs text-gray-400">
          Supports PNG, JPG, JPEG · 38 disease classes · 14 crop types
        </p>

      </div>

      {/* Floating AI assistant — always rendered so it persists across tab focus */}
      <PlantAssistant detectedDisease={detectedDisease} />
    </>
  );
}

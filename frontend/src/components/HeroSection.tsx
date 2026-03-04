"use client";

import Link from "next/link";
import { useState } from "react";
import { motion } from "framer-motion";
import { TextScramble } from "@/components/ui/text-scramble";
import { RevealWaveImage } from "@/components/ui/reveal-wave-image";

export function HeroSection() {
  const [ctaHovered, setCtaHovered] = useState(false);

  return (
    <section className="relative flex min-h-[calc(100vh-4rem)] flex-col items-center justify-center overflow-hidden bg-slate-950">

      {/* ── Step 1: background fades in (0 → 0.4s) ── */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.8, ease: "easeOut" }}
        className="absolute inset-0 z-0"
      >
        <RevealWaveImage
          src="/hero.jpg"
          waveSpeed={0.1}
          waveFrequency={0.6}
          waveAmplitude={0.3}
          revealRadius={0.32}
          revealSoftness={0.9}
          pixelSize={2}
          mouseRadius={0.35}
          className="h-full w-full"
        />
      </motion.div>

      {/* All overlays are pointer-events-none so mouse events reach the canvas */}

      {/* Dark tint */}
      <div className="pointer-events-none absolute inset-0 z-10 bg-slate-950/30" />
      {/* Soft green ambient glow */}
      <div className="pointer-events-none absolute inset-0 z-10 bg-[radial-gradient(ellipse_65%_50%_at_50%_55%,rgba(34,197,94,0.09),transparent_70%)]" />
      {/* Top fade */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-28 bg-gradient-to-b from-slate-950/65 to-transparent" />
      {/* Bottom fade */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 h-20 bg-gradient-to-t from-slate-950/50 to-transparent" />

      {/* ── Step 2: content rises up after background is visible ── */}
      <motion.div
        initial={{ opacity: 0, y: 36 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7, duration: 1.0, ease: "easeOut" }}
        className="relative z-20 flex flex-col items-center px-4 text-center sm:px-6"
      >
        {/* Badge */}
        <motion.span
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.85, duration: 0.7, ease: "easeOut" }}
          className="inline-flex items-center gap-2 rounded-full border border-green-500/30 bg-green-500/10 px-4 py-1.5 text-xs font-medium text-green-400 sm:text-sm"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
          <TextScramble
            as="span"
            duration={0.9}
            speed={0.025}
            characterSet="ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
          >
            AI-Powered Plant Disease Detection
          </TextScramble>
        </motion.span>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0, duration: 0.8, ease: "easeOut" }}
          className="mt-5 flex flex-col items-center gap-0 bg-gradient-to-br from-white via-green-100 to-green-300 bg-clip-text font-bold tracking-tight text-transparent sm:mt-6"
        >
          <TextScramble
            as="span"
            className="text-3xl sm:text-5xl lg:text-6xl"
            duration={1.0}
            speed={0.035}
          >
            Detect plant diseases
          </TextScramble>
          <TextScramble
            as="span"
            className="text-3xl sm:text-5xl lg:text-6xl"
            duration={1.0}
            speed={0.035}
            style={{ animationDelay: "200ms" }}
          >
            before they spread.
          </TextScramble>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2, duration: 0.7, ease: "easeOut" }}
          className="mx-auto mt-4 max-w-lg text-base text-gray-400 sm:mt-6 sm:max-w-xl sm:text-lg"
        >
          Upload a leaf image and get an accurate AI diagnosis in under a
          second —{" "}
          <span className="font-medium text-green-400">38 disease classes</span>{" "}
          across{" "}
          <span className="font-medium text-green-400">14 crop types</span>.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4, duration: 0.7, ease: "easeOut" }}
          className="mt-8 flex w-full flex-col items-center gap-3 sm:mt-10 sm:w-auto sm:flex-row sm:gap-4"
        >
          <Link
            href="/disease-recognition"
            onMouseEnter={() => setCtaHovered(true)}
            onMouseLeave={() => setCtaHovered(false)}
            className="w-full rounded-lg bg-green-600 px-7 py-3 text-center text-sm font-semibold text-white shadow-lg shadow-green-900/40 transition-[transform,box-shadow,background-color] duration-150 hover:-translate-y-0.5 hover:bg-green-500 hover:shadow-xl hover:shadow-green-900/50 active:translate-y-0 active:scale-[0.98] sm:w-auto"
          >
            <TextScramble
              as="span"
              duration={0.5}
              speed={0.02}
              characterSet="ABCDEFGHIJKLMNOPQRSTUVWXYZ→ "
              trigger={ctaHovered}
              onScrambleComplete={() => setCtaHovered(false)}
            >
              Analyze a Leaf →
            </TextScramble>
          </Link>

          <Link
            href="/about"
            className="w-full rounded-lg border border-white/15 bg-white/5 px-7 py-3 text-center text-sm font-semibold text-white backdrop-blur-sm transition-[transform,background-color] duration-150 hover:-translate-y-0.5 hover:bg-white/10 active:translate-y-0 sm:w-auto"
          >
            Learn More
          </Link>
        </motion.div>
      </motion.div>
    </section>
  );
}

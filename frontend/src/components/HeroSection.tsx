"use client";

import Link from "next/link";
import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { TextScramble } from "@/components/ui/text-scramble";
import { RevealWaveImage } from "@/components/ui/reveal-wave-image";
import { cn } from "@/lib/utils";

const HERO_IMAGES = ["/hero.jpg", "/hero2.jpg", "/hero3.jpg", "/hero4.jpg"];
// How long each image stays visible before the transition begins
const SLIDE_INTERVAL_MS = 7000;

const WAVE_PROPS = {
  waveSpeed: 0.1,
  waveFrequency: 0.6,
  waveAmplitude: 0.3,
  revealRadius: 0.32,
  revealSoftness: 0.9,
  pixelSize: 2,
  mouseRadius: 0.35,
} as const;

export function HeroSection() {
  const [ctaHovered, setCtaHovered] = useState(false);
  const [currentIdx, setCurrentIdx] = useState(0);

  const nextIdx = (currentIdx + 1) % HERO_IMAGES.length;

  const advance = useCallback(() => {
    setCurrentIdx((i) => (i + 1) % HERO_IMAGES.length);
  }, []);

  useEffect(() => {
    const timer = setInterval(advance, SLIDE_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [advance]);

  return (
    <section className="relative flex min-h-[calc(100vh-4rem)] flex-col items-center justify-center overflow-hidden bg-slate-950">

      {/* ── Slideshow ────────────────────────────────────────────────────────── */}
      <div className="absolute inset-0 z-0">

        {/* Active canvas — fades + very gently scales in */}
        <AnimatePresence mode="sync">
          <motion.div
            key={currentIdx}
            initial={{ opacity: 0, scale: 1.04 }}
            animate={{ opacity: 1, scale: 1.0 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={{
              opacity: { duration: 2.2, ease: "easeInOut" },
              scale:   { duration: 2.5, ease: "easeInOut" },
            }}
            className="absolute inset-0"
          >
            <RevealWaveImage
              src={HERO_IMAGES[currentIdx]}
              {...WAVE_PROPS}
              className="h-full w-full"
            />
          </motion.div>
        </AnimatePresence>

        {/* Next canvas pre-warmed — mounted but invisible so its WebGL context
            and texture are ready before it needs to appear. pointer-events-none
            so it doesn't intercept mouse events. */}
        <div
          key={`prewarm-${nextIdx}`}
          className="pointer-events-none absolute inset-0 opacity-0"
          aria-hidden
        >
          <RevealWaveImage
            src={HERO_IMAGES[nextIdx]}
            {...WAVE_PROPS}
            className="h-full w-full"
          />
        </div>
      </div>

      {/* ── Overlays (all pointer-events-none so mouse reaches the canvas) ──── */}

      {/* Dark tint — keeps text readable over the dithered bg */}
      <div className="pointer-events-none absolute inset-0 z-10 bg-slate-950/30" />

      {/* Soft green ambient glow */}
      <div className="pointer-events-none absolute inset-0 z-10 bg-[radial-gradient(ellipse_65%_50%_at_50%_55%,rgba(34,197,94,0.09),transparent_70%)]" />

      {/* Top fade — nav edge blends in */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-28 bg-gradient-to-b from-slate-950/65 to-transparent" />

      {/* Bottom fade — eases into the page content below */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 h-20 bg-gradient-to-t from-slate-950/50 to-transparent" />

      {/* ── Dot indicators ───────────────────────────────────────────────────── */}
      <div className="absolute bottom-6 left-1/2 z-20 flex -translate-x-1/2 gap-2">
        {HERO_IMAGES.map((_, i) => (
          <button
            key={i}
            onClick={() => setCurrentIdx(i)}
            aria-label={`Go to slide ${i + 1}`}
            className={cn(
              "h-1.5 rounded-full transition-all duration-500",
              i === currentIdx
                ? "w-6 bg-green-400"
                : "w-1.5 bg-white/25 hover:bg-white/50",
            )}
          />
        ))}
      </div>

      {/* ── Hero content ─────────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.9, ease: "easeOut" }}
        className="relative z-20 flex flex-col items-center px-4 text-center sm:px-6"
      >
        {/* Badge */}
        <span className="inline-flex items-center gap-2 rounded-full border border-green-500/30 bg-green-500/10 px-4 py-1.5 text-xs font-medium text-green-400 sm:text-sm">
          <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
          <TextScramble
            as="span"
            duration={0.9}
            speed={0.025}
            characterSet="ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
          >
            AI-Powered Plant Disease Detection
          </TextScramble>
        </span>

        {/* Headline */}
        <h1 className="mt-5 flex flex-col items-center gap-0 bg-gradient-to-br from-white via-green-100 to-green-300 bg-clip-text font-bold tracking-tight text-transparent sm:mt-6">
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
        </h1>

        {/* Subtitle */}
        <p className="mx-auto mt-4 max-w-lg text-base text-gray-400 sm:mt-6 sm:max-w-xl sm:text-lg">
          Upload a leaf image and get an accurate AI diagnosis in under a
          second —{" "}
          <span className="font-medium text-green-400">38 disease classes</span>{" "}
          across{" "}
          <span className="font-medium text-green-400">14 crop types</span>.
        </p>

        {/* CTAs */}
        <div className="mt-8 flex w-full flex-col items-center gap-3 sm:mt-10 sm:w-auto sm:flex-row sm:gap-4">
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
        </div>
      </motion.div>
    </section>
  );
}

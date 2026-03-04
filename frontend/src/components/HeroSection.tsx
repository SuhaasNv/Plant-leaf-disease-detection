"use client";

import Link from "next/link";
import { useState } from "react";
import { motion } from "framer-motion";
import { LampContainer } from "@/components/ui/lamp";
import { TextScramble } from "@/components/ui/text-scramble";

export function HeroSection() {
  const [ctaHovered, setCtaHovered] = useState(false);

  // -mt-16 pulls the hero behind the sticky header so backdrop-blur picks up dark bg
  return (
    <LampContainer className="-mt-16 rounded-none">
      <motion.div
        initial={{ opacity: 0, y: 60 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.8, ease: "easeInOut" }}
        className="flex flex-col items-center text-center"
      >
        {/* Badge — scrambles once on mount */}
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

        {/* Headline — two lines, each scrambles independently */}
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
            /* slight delay so second line starts after first */
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
          {/* Primary CTA — text scrambles on hover */}
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

          {/* Secondary CTA */}
          <Link
            href="/about"
            className="w-full rounded-lg border border-white/15 bg-white/5 px-7 py-3 text-center text-sm font-semibold text-white backdrop-blur-sm transition-[transform,background-color] duration-150 hover:-translate-y-0.5 hover:bg-white/10 active:translate-y-0 sm:w-auto"
          >
            Learn More
          </Link>
        </div>
      </motion.div>
    </LampContainer>
  );
}

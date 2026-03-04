"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const HERO_IMAGES = ["/hero.jpg", "/hero2.jpg", "/hero3.jpg", "/hero4.jpg"];
const SLIDE_INTERVAL_MS = 5000;
const FADE_DURATION_MS = 1200;

export const LampContainer = ({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) => {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [prevIdx, setPrevIdx] = useState<number | null>(null);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setFading(true);
      setPrevIdx(currentIdx);
      const next = (currentIdx + 1) % HERO_IMAGES.length;
      setCurrentIdx(next);
      // Clear the outgoing image after the crossfade completes
      const cleanup = setTimeout(() => {
        setPrevIdx(null);
        setFading(false);
      }, FADE_DURATION_MS);
      return () => clearTimeout(cleanup);
    }, SLIDE_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [currentIdx]);

  return (
    <div
      className={cn(
        // Shorter on mobile so content doesn't float off-screen
        "relative flex min-h-[85vh] w-full flex-col items-center justify-center overflow-hidden rounded-md bg-slate-950 sm:min-h-screen",
        className
      )}
    >
      {/* Background hero slideshow — crossfades between 4 images */}
      <div className="absolute inset-0 z-0">
        {/* Outgoing image — fades out */}
        {prevIdx !== null && fading && (
          <Image
            key={`prev-${prevIdx}`}
            src={HERO_IMAGES[prevIdx]}
            alt=""
            fill
            className="object-cover scale-105 blur-[3px] absolute inset-0 transition-opacity duration-[1200ms] ease-in-out opacity-0"
          />
        )}
        {/* Incoming / current image — fades in */}
        <Image
          key={`curr-${currentIdx}`}
          src={HERO_IMAGES[currentIdx]}
          alt=""
          fill
          priority={currentIdx === 0}
          className={cn(
            "object-cover scale-105 blur-[3px] absolute inset-0 transition-opacity ease-in-out",
            fading
              ? "opacity-100 duration-[1200ms]"
              : "opacity-100 duration-[1200ms]"
          )}
        />
        {/* Dark overlay so lamp glow and text remain the clear focus */}
        <div className="absolute inset-0 bg-slate-950/65" />
        {/* Slide indicator dots */}
        <div className="absolute bottom-6 left-1/2 z-10 flex -translate-x-1/2 gap-2">
          {HERO_IMAGES.map((_, i) => (
            <button
              key={i}
              onClick={() => setCurrentIdx(i)}
              aria-label={`Go to slide ${i + 1}`}
              className={cn(
                "h-1.5 rounded-full transition-all duration-500",
                i === currentIdx
                  ? "w-6 bg-green-400"
                  : "w-1.5 bg-white/30 hover:bg-white/50"
              )}
            />
          ))}
        </div>
      </div>

      {/* Lamp light cone */}
      <div className="relative z-0 flex w-full flex-1 scale-y-125 items-center justify-center isolate">

        {/* Left cone — green, narrower on mobile */}
        <motion.div
          initial={{ opacity: 0.5, width: "8rem" }}
          whileInView={{ opacity: 1, width: "18rem" }}
          transition={{ delay: 0.3, duration: 0.8, ease: "easeInOut" }}
          style={{
            backgroundImage: `conic-gradient(var(--conic-position), var(--tw-gradient-stops))`,
          }}
          className="absolute inset-auto right-1/2 h-56 overflow-visible bg-gradient-conic from-green-500 via-transparent to-transparent text-white [--conic-position:from_70deg_at_center_top] sm:w-[30rem]"
        >
          <div className="absolute bottom-0 left-0 z-20 h-40 w-full bg-slate-950 [mask-image:linear-gradient(to_top,white,transparent)]" />
          <div className="absolute bottom-0 left-0 z-20 h-full w-40 bg-slate-950 [mask-image:linear-gradient(to_right,white,transparent)]" />
        </motion.div>

        {/* Right cone — green, narrower on mobile */}
        <motion.div
          initial={{ opacity: 0.5, width: "8rem" }}
          whileInView={{ opacity: 1, width: "18rem" }}
          transition={{ delay: 0.3, duration: 0.8, ease: "easeInOut" }}
          style={{
            backgroundImage: `conic-gradient(var(--conic-position), var(--tw-gradient-stops))`,
          }}
          className="absolute inset-auto left-1/2 h-56 bg-gradient-conic from-transparent via-transparent to-green-500 text-white [--conic-position:from_290deg_at_center_top] sm:w-[30rem]"
        >
          <div className="absolute bottom-0 right-0 z-20 h-full w-40 bg-slate-950 [mask-image:linear-gradient(to_left,white,transparent)]" />
          <div className="absolute bottom-0 right-0 z-20 h-40 w-full bg-slate-950 [mask-image:linear-gradient(to_top,white,transparent)]" />
        </motion.div>

        {/* Dark base blur */}
        <div className="absolute top-1/2 h-48 w-full translate-y-12 scale-x-150 bg-slate-950 blur-2xl" />
        <div className="absolute top-1/2 z-50 h-48 w-full bg-transparent opacity-10 backdrop-blur-md" />

        {/* Wide ambient glow */}
        <div className="absolute inset-auto z-50 h-36 w-48 -translate-y-1/2 rounded-full bg-green-500 opacity-50 blur-3xl sm:w-[28rem]" />

        {/* Tight core glow */}
        <motion.div
          initial={{ width: "4rem" }}
          whileInView={{ width: "10rem" }}
          transition={{ delay: 0.3, duration: 0.8, ease: "easeInOut" }}
          className="absolute inset-auto z-30 h-36 -translate-y-24 rounded-full bg-green-400 blur-2xl sm:w-64"
        />

        {/* Horizontal light bar */}
        <motion.div
          initial={{ width: "8rem" }}
          whileInView={{ width: "18rem" }}
          transition={{ delay: 0.3, duration: 0.8, ease: "easeInOut" }}
          className="absolute inset-auto z-50 h-0.5 -translate-y-28 bg-green-400 sm:w-[30rem]"
        />

        {/* Dark cap */}
        <div className="absolute inset-auto z-40 h-44 w-full -translate-y-[12.5rem] bg-slate-950" />
      </div>

      {/* Hero content — position shifts up less on mobile */}
      <div className="relative z-50 -translate-y-40 px-4 sm:-translate-y-60 lg:-translate-y-80 sm:px-5">
        {children}
      </div>
    </div>
  );
};

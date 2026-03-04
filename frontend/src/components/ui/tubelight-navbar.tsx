"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export interface NavItem {
  name: string;
  url: string;
  icon: LucideIcon;
}

interface NavBarProps {
  items: NavItem[];
  className?: string;
}

/** Tubelight glow indicator that sits on top of the active pill */
export function TubelightGlow({ layoutId = "tubelight" }: { layoutId?: string }) {
  return (
    <motion.div
      layoutId={layoutId}
      className="absolute inset-0 -z-10 rounded-full bg-green-600/10"
      initial={false}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      {/* Bar */}
      <div className="absolute -top-2 left-1/2 -translate-x-1/2 h-1 w-8 rounded-t-full bg-green-600">
        {/* Glow layers */}
        <div className="absolute -left-2 -top-2 h-6 w-12 rounded-full bg-green-600/20 blur-md" />
        <div className="absolute -top-1 h-6 w-8 rounded-full bg-green-600/20 blur-md" />
        <div className="absolute left-2 top-0 h-4 w-4 rounded-full bg-green-600/20 blur-sm" />
      </div>
    </motion.div>
  );
}

/**
 * Standalone floating tubelight navbar.
 * Desktop: centered pill at top.
 * Mobile: icon bar pinned to the bottom.
 */
export function NavBar({ items, className }: NavBarProps) {
  const pathname = usePathname();

  return (
    <div
      className={cn(
        "fixed bottom-0 left-1/2 z-50 mb-4 -translate-x-1/2 sm:bottom-auto sm:top-0 sm:mb-0 sm:pt-4",
        className,
      )}
    >
      <div className="flex items-center gap-1 rounded-full border border-gray-200/60 bg-white/90 px-1.5 py-1.5 shadow-lg backdrop-blur-md dark:border-white/10 dark:bg-black/60">
        {items.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.url;

          return (
            <Link
              key={item.name}
              href={item.url}
              className={cn(
                "relative rounded-full px-5 py-1.5 text-sm font-semibold transition-colors",
                isActive
                  ? "text-green-600"
                  : "text-gray-500 hover:text-gray-900",
              )}
            >
              {/* Desktop: text label */}
              <span className="hidden sm:inline">{item.name}</span>
              {/* Mobile: icon only */}
              <span className="sm:hidden">
                <Icon size={20} strokeWidth={2} />
              </span>

              {isActive && <TubelightGlow />}
            </Link>
          );
        })}
      </div>
    </div>
  );
}

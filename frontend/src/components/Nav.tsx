"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { motion } from "framer-motion";
import { Home, Leaf, Info } from "lucide-react";
import { TubelightGlow, type NavItem } from "@/components/ui/tubelight-navbar";
import { cn } from "@/lib/utils";

const navItems: NavItem[] = [
  { name: "Home",   url: "/",                    icon: Home },
  { name: "Detect", url: "/disease-recognition", icon: Leaf },
  { name: "About",  url: "/about",               icon: Info },
];

export function Nav() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // On the home page the hero is dark (slate-950), so use dark-mode nav styles.
  // On all other pages the background is gray-50 (light), so use light-mode styles.
  const isHome = pathname === "/";

  return (
    <>
      {/* ── Sticky header (logo + desktop pill nav + CTA) ── */}
      <header
        className={cn(
          "sticky top-0 z-50 border-b backdrop-blur-xl transition-colors duration-300",
          isHome
            ? "border-white/10 bg-white/5"   // over dark hero
            : "border-gray-100 bg-white/80",  // over light pages
        )}
      >
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6 lg:px-8">

          {/* Logo */}
          <Link
            href="/"
            onClick={() => setMobileMenuOpen(false)}
            className={cn(
              "flex items-center gap-2.5 transition-opacity hover:opacity-80",
              isHome ? "text-white" : "text-gray-900",
            )}
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-green-600 text-white">
              🌿
            </span>
            <span className="font-semibold tracking-tight">LeafScan AI</span>
          </Link>

          {/* Desktop: tubelight pill nav */}
          <nav
            className={cn(
              "hidden items-center gap-0.5 rounded-full border px-1.5 py-1.5 shadow-sm backdrop-blur-md sm:flex",
              isHome
                ? "border-white/20 bg-white/10"
                : "border-gray-200/60 bg-gray-50/80",
            )}
          >
            {navItems.map(({ name, url }) => {
              const isActive = pathname === url;
              return (
                <Link
                  key={name}
                  href={url}
                  className={cn(
                    "relative rounded-full px-5 py-1.5 text-sm font-semibold transition-colors",
                    isActive
                      ? "text-green-400"
                      : isHome
                        ? "text-white/80 hover:text-white"
                        : "text-gray-600 hover:text-gray-900",
                  )}
                >
                  {name}
                  {isActive && <TubelightGlow layoutId="header-tubelight" />}
                </Link>
              );
            })}
          </nav>

          {/* Right: CTA + mobile hamburger */}
          <div className="flex items-center gap-2">
            <Link
              href="/disease-recognition"
              className="hidden rounded-lg bg-green-600 px-4 py-1.5 text-sm font-medium text-white transition-[transform,background-color] duration-150 hover:-translate-y-0.5 hover:bg-green-700 active:translate-y-0 sm:block"
            >
              Try It Free
            </Link>

            {/* Hamburger (mobile only) */}
            <button
              onClick={() => setMobileMenuOpen((o) => !o)}
              aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
              className={cn(
                "flex h-9 w-9 items-center justify-center rounded-lg transition-colors sm:hidden",
                isHome
                  ? "text-white/80 hover:bg-white/10"
                  : "text-gray-600 hover:bg-gray-100",
              )}
            >
              {mobileMenuOpen ? (
                <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M6 6l12 12M6 18L18 6" />
                </svg>
              ) : (
                <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile dropdown */}
        {mobileMenuOpen && (
          <div
            className={cn(
              "border-t px-4 pb-4 pt-2 backdrop-blur-xl sm:hidden",
              isHome
                ? "border-white/10 bg-slate-900/80"
                : "border-gray-100 bg-white/90",
            )}
          >
            <nav className="flex flex-col gap-1">
              {navItems.map(({ name, url }) => (
                <Link
                  key={name}
                  href={url}
                  onClick={() => setMobileMenuOpen(false)}
                  className={cn(
                    "rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                    pathname === url
                      ? isHome
                        ? "bg-green-900/40 text-green-400"
                        : "bg-green-50 text-green-600"
                      : isHome
                        ? "text-white/70 hover:bg-white/10 hover:text-white"
                        : "text-gray-600 hover:bg-gray-50 hover:text-gray-900",
                  )}
                >
                  {name}
                </Link>
              ))}
              <Link
                href="/disease-recognition"
                onClick={() => setMobileMenuOpen(false)}
                className="mt-2 rounded-lg bg-green-600 px-4 py-2.5 text-center text-sm font-semibold text-white transition-colors hover:bg-green-700"
              >
                Try It Free
              </Link>
            </nav>
          </div>
        )}
      </header>

      {/* ── Mobile bottom tubelight bar (icon only, above page fold) ── */}
      <div className="fixed bottom-0 left-1/2 z-50 mb-4 -translate-x-1/2 sm:hidden">
        <div className="flex items-center gap-1 rounded-full border border-white/20 bg-white/15 px-2 py-2 shadow-xl backdrop-blur-xl">
          {navItems.map(({ name, url, icon: Icon }) => {
            const isActive = pathname === url;
            return (
              <Link
                key={name}
                href={url}
                aria-label={name}
                className={cn(
                  "relative rounded-full p-3 transition-colors",
                  isActive ? "text-green-600" : "text-gray-400 hover:text-gray-700",
                )}
              >
                <Icon size={20} strokeWidth={2} />
                {isActive && <TubelightGlow layoutId="bottom-tubelight" />}
              </Link>
            );
          })}
        </div>
      </div>
    </>
  );
}

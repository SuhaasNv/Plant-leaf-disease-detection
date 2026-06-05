"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Home, Leaf, Info, Menu, X } from "lucide-react";
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
  // Scrolled state — nav gets a darker/more opaque bg when user scrolls down
  const [scrolled, setScrolled] = useState(false);

  const isHome = pathname === "/";

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Close mobile menu on route change
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMobileMenuOpen(false);
  }, [pathname]);

  return (
    <>
      <header
        className={cn(
          "sticky top-0 z-50 transition-all duration-300",
          isHome
            // Dark hero page — always dark so logo/links are always legible
            ? scrolled
              ? "bg-slate-950/90 backdrop-blur-xl border-b border-white/10 shadow-lg shadow-black/20"
              : "bg-slate-950/75 backdrop-blur-xl border-b border-white/8"
            // Light pages — white with subtle shadow
            : "bg-white/90 backdrop-blur-xl border-b border-gray-100 shadow-sm",
        )}
      >
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6 lg:px-8">

          {/* Logo */}
          <Link
            href="/"
            className={cn(
              "flex items-center gap-2.5 transition-opacity hover:opacity-80",
              isHome ? "text-white" : "text-gray-900",
            )}
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-green-600 text-white text-base shadow-md shadow-green-900/30">
              🌿
            </span>
            <span className="font-semibold tracking-tight">LeafScan AI</span>
          </Link>

          {/* Desktop nav pill */}
          <nav
            className={cn(
              "hidden items-center gap-0.5 rounded-full border px-1.5 py-1.5 shadow-sm backdrop-blur-md sm:flex",
              isHome
                ? "border-white/15 bg-white/10"
                : "border-gray-200/70 bg-gray-50/80",
            )}
          >
            {navItems.map(({ name, url }) => {
              const isActive = pathname === url;
              return (
                <Link
                  key={name}
                  href={url}
                  className={cn(
                    "relative rounded-full px-5 py-1.5 text-sm font-semibold transition-colors duration-150",
                    isActive
                      ? "text-green-400"
                      : isHome
                        ? "text-white/85 hover:text-white"
                        : "text-gray-600 hover:text-gray-900",
                  )}
                >
                  {name}
                  {isActive && <TubelightGlow layoutId="header-tubelight" />}
                </Link>
              );
            })}
          </nav>

          {/* Right side */}
          <div className="flex items-center gap-2">
            <Link
              href="/disease-recognition"
              className="hidden rounded-lg bg-green-600 px-4 py-1.5 text-sm font-semibold text-white shadow-md shadow-green-900/30 transition-[transform,background-color,box-shadow] duration-150 hover:-translate-y-0.5 hover:bg-green-500 hover:shadow-lg active:translate-y-0 sm:block"
            >
              Try It Free
            </Link>

            {/* Hamburger */}
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
              {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile dropdown */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2, ease: "easeInOut" }}
              className="overflow-hidden"
            >
              <div
                className={cn(
                  "border-t px-4 pb-4 pt-2 sm:hidden",
                  isHome
                    ? "border-white/10 bg-slate-900/90 backdrop-blur-xl"
                    : "border-gray-100 bg-white/95",
                )}
              >
                <nav className="flex flex-col gap-1">
                  {navItems.map(({ name, url, icon: Icon }) => (
                    <Link
                      key={name}
                      href={url}
                      className={cn(
                        "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                        pathname === url
                          ? isHome
                            ? "bg-green-900/40 text-green-400"
                            : "bg-green-50 text-green-600"
                          : isHome
                            ? "text-white/70 hover:bg-white/10 hover:text-white"
                            : "text-gray-600 hover:bg-gray-50 hover:text-gray-900",
                      )}
                    >
                      <Icon size={16} strokeWidth={2} className="shrink-0" />
                      {name}
                    </Link>
                  ))}
                  <Link
                    href="/disease-recognition"
                    className="mt-2 flex items-center justify-center gap-2 rounded-lg bg-green-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-green-700"
                  >
                    <Leaf size={15} />
                    Try It Free
                  </Link>
                </nav>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* Mobile bottom bar */}
      <div className="fixed bottom-0 left-1/2 z-50 mb-4 -translate-x-1/2 sm:hidden">
        <div
          className={cn(
            "flex items-center gap-1 rounded-full border px-2 py-2 shadow-xl backdrop-blur-xl",
            isHome
              ? "border-white/15 bg-slate-900/80"
              : "border-gray-200/60 bg-white/90",
          )}
        >
          {navItems.map(({ name, url, icon: Icon }) => {
            const isActive = pathname === url;
            return (
              <Link
                key={name}
                href={url}
                aria-label={name}
                className={cn(
                  "relative rounded-full p-3 transition-colors",
                  isActive
                    ? "text-green-500"
                    : isHome
                      ? "text-white/50 hover:text-white/80"
                      : "text-gray-400 hover:text-gray-700",
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

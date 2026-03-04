import type { Metadata } from "next";
import { DM_Sans } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/Nav";

const dmSans = DM_Sans({
  variable: "--font-dm-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Plant Disease Recognition | AI-Powered Leaf Analysis",
  description:
    "Identify plant diseases from leaf images using advanced machine learning. Protect your crops with accurate, fast disease detection.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${dmSans.variable} min-h-screen antialiased`}>
        <div className="flex min-h-screen flex-col">
          <Nav />
          {/* pb-20 on mobile reserves space above the fixed bottom nav bar */}
          <main className="flex-1 pb-20 sm:pb-0">{children}</main>
        </div>
      </body>
    </html>
  );
}

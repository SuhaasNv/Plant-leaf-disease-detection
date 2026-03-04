import Link from "next/link";
import { HeroSection } from "@/components/HeroSection";

const steps = [
  {
    number: "01",
    title: "Upload a Leaf Image",
    description:
      "Take a photo of any plant leaf and upload it directly from your device. Supports PNG, JPG, and JPEG.",
  },
  {
    number: "02",
    title: "AI Analysis",
    description:
      "Our CNN model processes the image in milliseconds, scanning for visual patterns associated with 38 disease classes.",
  },
  {
    number: "03",
    title: "Instant Results",
    description:
      "Receive a clear diagnosis with the disease name and confidence score — ready to act on immediately.",
  },
];

const features = [
  "~95% accuracy across 38 disease classes",
  "Results in under 1 second",
  "14 crop types supported",
  "State-of-the-art CNN architecture",
  "Simple, intuitive interface",
  "No account required",
];

const stats = [
  { value: "95%", label: "Accuracy" },
  { value: "38",  label: "Disease Classes" },
  { value: "14",  label: "Crop Types" },
  { value: "<1s", label: "Inference Time" },
];

export default function Home() {
  return (
    <>
      {/* ── Hero ── */}
      <HeroSection />

      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 sm:py-16 lg:px-8">

        {/* ── Stats ── */}
        <section>
          <div className="rounded-2xl bg-white shadow-xl">
            <dl className="grid grid-cols-2 divide-x divide-y divide-gray-100 sm:grid-cols-4 sm:divide-y-0">
              {stats.map((s) => (
                <div key={s.label} className="px-4 py-6 text-center sm:px-8 sm:py-8">
                  <dt className="text-3xl font-bold text-green-600 sm:text-4xl">{s.value}</dt>
                  <dd className="mt-1 text-xs text-gray-600 sm:mt-1.5 sm:text-sm">{s.label}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        {/* ── How it works ── */}
        <section className="mt-16 sm:mt-24">
          <div className="text-center">
            <h2 className="text-2xl font-bold tracking-tight text-gray-900 sm:text-3xl">
              How it works
            </h2>
            <p className="mt-2 text-gray-600 sm:mt-3">
              Three steps from upload to diagnosis.
            </p>
          </div>

          <div className="mt-8 grid gap-4 sm:mt-12 sm:grid-cols-3 sm:gap-6">
            {steps.map((step) => (
              <div
                key={step.number}
                className="rounded-2xl bg-white p-6 shadow-xl transition-[transform,box-shadow] duration-200 hover:-translate-y-1 hover:shadow-2xl sm:p-8"
              >
                <span className="text-4xl font-bold text-gray-100 sm:text-5xl">
                  {step.number}
                </span>
                <h3 className="mt-3 font-semibold text-gray-900 sm:mt-4">
                  {step.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-gray-600">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Features ── */}
        <section className="mt-16 sm:mt-24">
          <div className="text-center">
            <h2 className="text-2xl font-bold tracking-tight text-gray-900 sm:text-3xl">
              Why LeafScan AI
            </h2>
            <p className="mt-2 text-gray-600 sm:mt-3">
              Built for accuracy. Designed for simplicity.
            </p>
          </div>

          <ul className="mt-6 grid gap-3 sm:mt-10 sm:grid-cols-2 lg:grid-cols-3">
            {features.map((f) => (
              <li
                key={f}
                className="flex items-center gap-3 rounded-xl bg-white px-4 py-4 shadow-md transition-[transform,box-shadow] duration-200 hover:-translate-y-0.5 hover:shadow-lg sm:px-5"
              >
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-green-100 text-xs font-bold text-green-600">
                  ✓
                </span>
                <span className="text-sm text-gray-700">{f}</span>
              </li>
            ))}
          </ul>
        </section>

        {/* ── CTA ── */}
        <section className="mt-16 sm:mt-24">
          <div className="rounded-2xl bg-green-600 px-6 py-10 text-center shadow-xl sm:px-8 sm:py-16">
            <h2 className="text-2xl font-bold text-white sm:text-3xl">
              Ready to protect your crops?
            </h2>
            <p className="mt-3 text-sm text-green-100 sm:text-base">
              Upload a leaf photo and get an AI diagnosis in seconds. No sign-up needed.
            </p>
            <Link
              href="/disease-recognition"
              className="mt-6 inline-flex items-center gap-2 rounded-lg bg-white px-7 py-3 text-sm font-semibold text-green-600 shadow transition-[transform,background-color] duration-150 hover:-translate-y-0.5 hover:bg-green-50 active:translate-y-0 sm:mt-8 sm:px-8"
            >
              Start Detection
              <span aria-hidden>→</span>
            </Link>
          </div>
        </section>

      </div>
    </>
  );
}

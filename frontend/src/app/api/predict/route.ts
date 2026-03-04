/**
 * Server-side proxy for the plant-disease prediction API.
 *
 * Why a proxy?
 *  - API_KEY and API_URL never appear in the browser bundle (no NEXT_PUBLIC_ prefix).
 *  - We can enforce a second layer of rate limiting close to the user (Vercel edge).
 *  - Frontend always calls /api/predict — completely decoupled from the Railway URL.
 *
 * Required env vars (Vercel dashboard → Settings → Environment Variables):
 *   API_URL   e.g. https://plant-leaf-disease-detection-production-7a13.up.railway.app
 *   API_KEY   must match the API_KEY set on Railway
 *
 * Optional:
 *   PROXY_RATE_LIMIT_RPM  requests per minute per IP (default 10)
 */

import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL ?? "";
const API_KEY = process.env.API_KEY ?? "";

// ── Simple in-memory rate limiter ─────────────────────────────────────────────
// Resets on each Vercel cold-start, which is acceptable for a portfolio project.
// For production, replace with Vercel KV or Upstash Redis.
const RPM = parseInt(process.env.PROXY_RATE_LIMIT_RPM ?? "10", 10);
const _windowMs = 60_000;
const _hits = new Map<string, { count: number; resetAt: number }>();

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  const entry = _hits.get(ip);
  if (!entry || now > entry.resetAt) {
    _hits.set(ip, { count: 1, resetAt: now + _windowMs });
    return false;
  }
  entry.count += 1;
  return entry.count > RPM;
}

// ── Client-side input guards ──────────────────────────────────────────────────
const MAX_BYTES = 10 * 1024 * 1024; // 10 MB — mirrors backend limit

export async function POST(req: NextRequest) {
  // ── Misconfiguration guard ────────────────────────────────────────────────
  if (!API_URL) {
    return NextResponse.json(
      { detail: "API_URL is not configured on the server." },
      { status: 503 }
    );
  }

  // ── Rate limiting ─────────────────────────────────────────────────────────
  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0].trim() ?? "unknown";
  if (isRateLimited(ip)) {
    return NextResponse.json(
      { detail: "Too many requests. Please wait a moment and try again." },
      { status: 429 }
    );
  }

  // ── Parse and validate the multipart body ────────────────────────────────
  let formData: FormData;
  try {
    formData = await req.formData();
  } catch {
    return NextResponse.json(
      { detail: "Invalid request body. Expected multipart/form-data." },
      { status: 400 }
    );
  }

  const file = formData.get("file");
  if (!(file instanceof File)) {
    return NextResponse.json(
      { detail: 'Missing "file" field in the form data.' },
      { status: 400 }
    );
  }

  if (file.size > MAX_BYTES) {
    return NextResponse.json(
      { detail: `File too large. Maximum allowed size is ${MAX_BYTES / 1_048_576} MB.` },
      { status: 413 }
    );
  }

  const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/gif"];
  if (!allowedTypes.includes(file.type)) {
    return NextResponse.json(
      { detail: "Unsupported file type. Please upload a JPEG, PNG, or WebP image." },
      { status: 400 }
    );
  }

  // ── Forward to Railway backend ────────────────────────────────────────────
  const upstream = new FormData();
  upstream.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
      },
      body: upstream,
    });
  } catch (err) {
    console.error("[predict proxy] upstream fetch failed:", err);
    return NextResponse.json(
      { detail: "Cannot reach the prediction service. Please try again later." },
      { status: 502 }
    );
  }

  // ── Return upstream response verbatim ────────────────────────────────────
  const data = await res.json().catch(() => ({ detail: "Unexpected response from prediction service." }));
  return NextResponse.json(data, { status: res.status });
}

/**
 * Server-side proxy for the /chat endpoint.
 * Keeps API_URL out of the browser bundle.
 */
import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL ?? "";

export async function POST(req: NextRequest) {
  if (!API_URL) {
    return NextResponse.json(
      { detail: "API_URL is not configured on the server." },
      { status: 503 }
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json(
      { detail: "Invalid JSON body." },
      { status: 400 }
    );
  }

  let res: Response;
  try {
    res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    console.error("[chat proxy] upstream fetch failed:", err);
    return NextResponse.json(
      { detail: "Cannot reach the prediction service. Please try again later." },
      { status: 502 }
    );
  }

  const data = await res
    .json()
    .catch(() => ({ detail: "Unexpected response from chat service." }));
  return NextResponse.json(data, { status: res.status });
}

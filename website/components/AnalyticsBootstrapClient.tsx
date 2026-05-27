"use client";

if (typeof window !== "undefined") {
  import("@/lib/instrumentation-client").catch((err) => {
    console.error("Instrumentation bootstrap failed", err);
  });
}

export default function AnalyticsBootstrapClient() {
  return null;
}

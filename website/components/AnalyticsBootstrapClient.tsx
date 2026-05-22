"use client";

if (typeof window !== "undefined") {
  import("@/lib/instrumentation-client").catch(() => {
    // ignore
  });
}

export default function AnalyticsBootstrapClient() {
  return null;
}

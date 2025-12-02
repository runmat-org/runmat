"use client";

import { useEffect } from "react";

export default function AnalyticsBootstrap() {
  useEffect(() => {
    // Dynamically import to ensure client-side execution
    import("@/lib/instrumentation-client").catch(() => {
      // ignore
    });
  }, []);
  return null;
}
"use client";

import { useEffect } from "react";

export default function AnalyticsBootstrapClient() {
  useEffect(() => {
    import("@/lib/instrumentation-client").catch(() => {
      // ignore
    });
  }, []);

  return null;
}

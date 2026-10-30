/** Runtime-only headers. Never put these values in runtimeSettings or saved project/user documents. */
export type OpenTelemetryStartupHeaders = Readonly<Record<string, string>>;

/** Collector acknowledgement counts, not downstream indexing/storage confirmation. */
export interface OpenTelemetryExportStats {
  droppedRecords: number;
  failedBatches: number;
  exportedSpans: number;
  exportedLogs: number;
  collectorWarnings: number;
  queuedBytes: number;
}

/** Native generated exports used by the SDK's public asynchronous wrappers. */
export interface OpenTelemetryNativeExports {
  sendOpenTelemetryTest?: () => Promise<OpenTelemetryExportStats>;
  flushOpenTelemetry?: () => Promise<OpenTelemetryExportStats>;
  shutdownOpenTelemetry?: () => Promise<OpenTelemetryExportStats>;
  openTelemetryStats?: () => OpenTelemetryExportStats;
}

export type OpenTelemetryOperation = keyof OpenTelemetryNativeExports;

export async function invokeOpenTelemetry(
  native: OpenTelemetryNativeExports,
  operation: OpenTelemetryOperation,
): Promise<OpenTelemetryExportStats> {
  const method: ((this: OpenTelemetryNativeExports) => OpenTelemetryExportStats | Promise<OpenTelemetryExportStats>) | undefined = native[operation];
  if (typeof method !== "function") {
    throw new Error(`The loaded RunMat module does not support ${operation}; rebuild the browser runtime assets.`);
  }
  // Do not turn a rejected collector request into a queued-success receipt.
  return await method.call(native);
}

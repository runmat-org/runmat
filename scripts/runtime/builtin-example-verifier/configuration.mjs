export function resolveTimeoutMs() {
    const raw = process.env.RUNMAT_EXAMPLE_TIMEOUT_MS;
    if (!raw) {
        return 15000;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return 15000;
    }
    return Math.floor(parsed);
}
export function resolveNativeTimeoutMs(browserTimeoutMs) {
    const raw = process.env.RUNMAT_EXAMPLE_NATIVE_TIMEOUT_MS;
    if (!raw) {
        return Math.max(60000, browserTimeoutMs);
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return Math.max(60000, browserTimeoutMs);
    }
    return Math.floor(parsed);
}

export function resolveConcurrency() {
    const raw = process.env.RUNMAT_EXAMPLE_CONCURRENCY;
    if (!raw) {
        return 4;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return 4;
    }
    return Math.floor(parsed);
}

export function resolveOverallTimeoutMs(perCaseMs, concurrency, totalCases) {
    const raw = process.env.RUNMAT_EXAMPLE_TOTAL_TIMEOUT_MS;
    if (raw) {
        const parsed = Number(raw);
        if (Number.isFinite(parsed) && parsed > 0) {
            return Math.floor(parsed);
        }
    }
    const safeConcurrency = Math.max(1, concurrency);
    const batches = Math.ceil(totalCases / safeConcurrency);
    const estimated = batches * perCaseMs + 120000; // Increased buffer
    return Math.max(1200000, estimated);
}

export function resolveLogIntervalMs() {
    const raw = process.env.RUNMAT_EXAMPLE_LOG_INTERVAL_MS;
    if (!raw) {
        return 2000;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0) {
        return 2000;
    }
    return Math.floor(parsed);
}

// @ts-check

import { spawnSync } from "node:child_process";
import { sha256 } from "./identity.mjs";

export const RUNTIME_FEATURE_PROBE_SCHEMA = "runmat.builtin-examples.runtime-feature-probe.v1";

export function probeNativeRuntimeFeatures(binary, artifactManifestDigest, invoke = spawnSync) {
    const completed = invoke(binary, ["--color=never", "info", "--json"], {
        encoding: "utf8", timeout: 30_000, maxBuffer: 256 * 1024,
        env: { ...process.env, NO_COLOR: "1" }
    });
    if (completed.error || completed.status !== 0) {
        throw new Error(`staged native runtime feature probe failed: ${completed.error?.message ?? completed.stderr?.trim() ?? completed.status}`);
    }
    const report = parseRuntimeFeatureReport(completed.stdout);
    const reportDigest = sha256(report);
    return {
        schema: RUNTIME_FEATURE_PROBE_SCHEMA,
        artifactManifestDigest,
        report,
        reportDigest,
        probeDigest: sha256([RUNTIME_FEATURE_PROBE_SCHEMA, artifactManifestDigest, reportDigest])
    };
}

export function parseRuntimeFeatureReport(encoded) {
    let report;
    try { report = JSON.parse(encoded); }
    catch { throw new Error("staged native runtime feature report is not JSON"); }
    exact(report, ["schema", "runtime"], "runtime feature report");
    if (report.schema !== "runmat.runtime-features.v1") throw new Error("unsupported staged runtime feature report schema");
    exact(report.runtime, ["crate_name", "schema_version", "known_features", "enabled_features"], "runtime feature inventory");
    if (report.runtime.crate_name !== "runmat-runtime" || report.runtime.schema_version !== 1) {
        throw new Error("staged product did not report canonical runmat-runtime features");
    }
    for (const key of ["known_features", "enabled_features"]) {
        const values = report.runtime[key];
        if (!Array.isArray(values) || values.some((value) => typeof value !== "string" || !value) || new Set(values).size !== values.length) {
            throw new Error(`invalid staged runtime ${key}`);
        }
    }
    if (!report.runtime.known_features.includes("plot-core")
        || report.runtime.enabled_features.some((feature) => !report.runtime.known_features.includes(feature))) {
        throw new Error("staged runtime feature inventory is inconsistent");
    }
    return report;
}

export function validateRuntimeFeatureProbe(probe, artifactManifestDigest) {
    exact(probe, ["schema", "artifactManifestDigest", "report", "reportDigest", "probeDigest"], "runtime feature probe");
    if (probe.schema !== RUNTIME_FEATURE_PROBE_SCHEMA || probe.artifactManifestDigest !== artifactManifestDigest) {
        throw new Error("runtime feature probe is not bound to the staged product");
    }
    parseRuntimeFeatureReport(JSON.stringify(probe.report));
    if (probe.reportDigest !== sha256(probe.report)
        || probe.probeDigest !== sha256([RUNTIME_FEATURE_PROBE_SCHEMA, artifactManifestDigest, probe.reportDigest])) {
        throw new Error("runtime feature probe digest mismatch");
    }
    return probe;
}

export function runtimeRequirementAvailability(requirements, fixture, probe) {
    for (const capability of requirements.runtime) {
        if (capability === "PlotCore") {
            if (!probe.report.runtime.enabled_features.includes("plot-core")) {
                return { available: false, reason: "the staged native runtime does not include plot-core" };
            }
            continue;
        }
        // Foreign fixtures perform their own artifact preparation and host
        // adapter probes. A compile-time runtime feature cannot prove JVM,
        // CPython, NumPy, or native-loader availability on the current host.
        const preparation = fixture?.ForeignAdapter?.preparation;
        const kind = preparation && Object.keys(preparation)[0];
        const adapterCapability = new Map([
            ["Java", "JavaVirtualMachine"], ["Python", "Python"],
            ["Mex", "Mex"], ["NativeFfi", "NativeFfi"]
        ]).get(kind ?? "");
        if (capability === adapterCapability) continue;
        return { available: false, reason: `runtime capability ${capability} has no staged-product availability probe` };
    }
    return { available: true, reason: "" };
}

function exact(value, keys, label) {
    if (!value || typeof value !== "object" || Array.isArray(value)
        || JSON.stringify(Object.keys(value).sort()) !== JSON.stringify([...keys].sort())) {
        throw new Error(`${label} has invalid fields`);
    }
}

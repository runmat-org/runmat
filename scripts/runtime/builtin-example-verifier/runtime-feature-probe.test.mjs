import assert from "node:assert/strict";
import test from "node:test";

import {
    probeNativeRuntimeFeatures, runtimeRequirementAvailability, validateRuntimeFeatureProbe
} from "./runtime-feature-probe.mjs";

const ARTIFACT = `sha256:${"a".repeat(64)}`;
const REPORT = {
    schema: "runmat.runtime-features.v1",
    runtime: {
        crate_name: "runmat-runtime", schema_version: 1,
        known_features: ["plot-core", "wgpu"], enabled_features: ["plot-core"]
    }
};

function probe(report = REPORT) {
    return probeNativeRuntimeFeatures("/staged/runmat", ARTIFACT, (binary, args) => {
        assert.equal(binary, "/staged/runmat");
        assert.deepEqual(args, ["--color=never", "info", "--json"]);
        return { status: 0, stdout: JSON.stringify(report), stderr: "" };
    });
}

test("compiled plot-core is admitted only from the staged product probe", () => {
    const observed = probe();
    assert.equal(validateRuntimeFeatureProbe(observed, ARTIFACT), observed);
    assert.deepEqual(runtimeRequirementAvailability({ runtime: ["PlotCore"] }, null, observed), {
        available: true, reason: ""
    });
    const withoutPlot = probe({ ...REPORT, runtime: { ...REPORT.runtime, enabled_features: [] } });
    assert.match(runtimeRequirementAvailability({ runtime: ["PlotCore"] }, null, withoutPlot).reason, /does not include plot-core/);
    assert.throws(() => validateRuntimeFeatureProbe(observed, `sha256:${"b".repeat(64)}`), /not bound/);
    assert.throws(() => validateRuntimeFeatureProbe({ ...observed, report: withoutPlot.report }, ARTIFACT), /digest mismatch/);
});

test("invalid feature reports and unproven host requirements fail closed", () => {
    assert.throws(() => probe({ ...REPORT, runtime: { ...REPORT.runtime, known_features: [] } }), /inconsistent/);
    const observed = probe();
    assert.match(runtimeRequirementAvailability({ runtime: ["NumPy"] }, null, observed).reason, /no staged-product availability probe/);
    assert.deepEqual(runtimeRequirementAvailability(
        { runtime: ["Python"] }, { ForeignAdapter: { preparation: { Python: {} } } }, observed
    ), { available: true, reason: "" });
});

use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/stepinfo.rs"),
};
const ADMISSION_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Stepinfo admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/stepinfo/admission.rs"),
};
const METRICS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Stepinfo metrics",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/stepinfo/metrics.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("stepinfo"), slug: Some("stepinfo"),
    summary: "Summarize a SISO step response or sampled response data.",
    description: "`stepinfo(sys)` simulates a supported real SISO `tf` model; `stepinfo(y)`, `stepinfo(y,t)` and `stepinfo(y,t,yfinal)` analyze supplied real samples. The result is a scalar structure with double-valued timing, extrema and percentage fields. Current model support is bounded to the response simulator's real SISO, no-delay subset; other LTI families and model arrays are not implied.",
    keywords: &["stepinfo", "step response", "rise time", "settling time", "overshoot", "control system"],
    related: &["step", "dcgain", "tf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Metrics and options", paragraphs: &[
            "The result contains RiseTime, TransientTime, SettlingTime, SettlingMin, SettlingMax, Overshoot, Undershoot, Peak, PeakTime and SteadyStateValue. Rise crossings are interpolated between adjacent samples. A response that does not settle returns a NaN settling time. Model-form values come from a sampled response rather than an analytic step solution.",
            "The default settling band is 2%. The supported name-value option `SettlingTimeThreshold` accepts a finite scalar strictly between zero and one; other public stepinfo options are not yet implemented. Supplied response and time vectors must have equal nonzero lengths; time samples must be finite and nondecreasing. An explicit yfinal must be a finite real scalar.",
        ] },
        BuiltinDocumentationSection { heading: "Numeric storage and execution", paragraphs: &[
            "Native typed-integer sampled y, t and yfinal are one RunMat-only extension; native typed-integer SettlingTimeThreshold is independently gated. In model form, typed-integer tf coefficients or Ts and logical tf coefficients assigned after construction use their model-property gates. Every admitted integer must be exactly representable in binary64 before floating-point metrics are calculated. Logical sampled data are interpreted as zero or one. Complex response data are not admitted.",
            "The result is a host scalar structure. Nested GPU-resident values in consumed model properties and sampled roles are gathered through their owning provider after host-known form, option and compatibility checks; unrelated model properties are not read. There is no stepinfo GPU kernel or fusion. Calls may suspend and throw, but perform no filesystem, network, workspace, callback, randomness or UI effect.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "model", title: "Metrics from a transfer function", program: "H = tf(1, [1 1]); info = stepinfo(H); fprintf('ans = %g\\n', info.Overshoot);", display_output: Some("ans = 0"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(info.SteadyStateValue - 1) < 1e-8); assert(info.Overshoot == 0);" } },
        BuiltinExample { id: "samples", title: "Analyze supplied samples", program: "y = [0 .5 .9 1]; t = [0 1 2 3]; info = stepinfo(y, t, 1); fprintf('ans = %g\\n', info.SteadyStateValue);", display_output: Some("ans = 1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(info.SteadyStateValue == 1); assert(abs(info.RiseTime - 1.8) < 1e-8);" } },
        BuiltinExample { id: "threshold", title: "Select a settling band", program: "y = [0 .5 .9 1]; t = [0 1 2 3]; info = stepinfo(y, t, 1, 'SettlingTimeThreshold', .1); fprintf('ans = %g\\n', info.SettlingTime);", display_output: Some("ans = 2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(info.SettlingTime == 2);" } },
        BuiltinExample { id: "feedback", title: "Metrics of a feedback model", program: "s = tf('s'); T = feedback(2/(s + 1), 1); info = stepinfo(T); fprintf('ans = %g\\n', info.Overshoot);", display_output: Some("ans = 0"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(info.Overshoot == 0); assert(abs(info.SteadyStateValue - 2/3) < 1e-8);" } },
        BuiltinExample { id: "exponential-samples", title: "Metrics of a sampled exponential response", program: "t = 0:0.1:5; y = 1 - exp(-t); info = stepinfo(y, t, 1); fprintf('ans = %g\\n', info.SteadyStateValue);", display_output: Some("ans = 1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(info.SteadyStateValue == 1); assert(info.Overshoot == 0);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does stepinfo require a model?", answer: "No. It accepts a real sampled response vector, optional time samples and an optional steady-state value." },
        BuiltinDocumentationFaq { question: "Are model-form metrics analytic?", answer: "No. They are calculated from a sampled step response, with linear interpolation of rise crossings." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "step", target: BuiltinDocumentationLinkTarget::Builtin("step") },
        BuiltinDocumentationLink { label: "dcgain", target: BuiltinDocumentationLinkTarget::Builtin("dcgain") },
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        SOURCE,
        ADMISSION_SOURCE,
        METRICS_SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE, ADMISSION_SOURCE, METRICS_SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Sampled/model forms, metrics, options and integer boundaries", location: "crates/runmat-runtime/src/builtins/control/stepinfo.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM control workflows", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["Model-form response numerics and native/browser/provider qualification require the integrated control batch."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

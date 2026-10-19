use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "timeit-1", title: "Time a simple anonymous function", program: "f = @() sin(rand(1000, 1));\nt = timeit(f)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(t, 'double') && isscalar(t) && isfinite(t) && t >= 0);" } },
    BuiltinExample { id: "timeit-2", title: "Compare two implementations", program: "A = rand(1e5, 1);\nslow = @() sum(A .* A);\nfast = @() sumsq(A);\nslowTime = timeit(slow);\nfastTime = timeit(fast)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(all(isfinite([slowTime fastTime])) && all([slowTime fastTime] >= 0));" } },
    BuiltinExample { id: "timeit-3", title: "Time a function with no requested outputs", program: "logMessage = @() fprintf('Iteration complete\\n');\nt = timeit(logMessage, 0)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(t) && isfinite(t) && t >= 0);" } },
    BuiltinExample { id: "timeit-4", title: "Request multiple callback outputs", program: "svdTime = timeit(@() svd(rand(256)), 3)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(svdTime) && isfinite(svdTime) && svdTime >= 0);" } },
    BuiltinExample { id: "timeit-5", title: "Measure GPU-bound work", program: "gfun = @() gather(sin(gpuArray.rand(4096, 1)));\ntgpu = timeit(gfun)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(tgpu) && isfinite(tgpu) && tgpu >= 0);" } },
    BuiltinExample { id: "timeit-6", title: "Time a preallocation helper", program: "makeMatrix = @() zeros(2048, 2048);\nt = timeit(makeMatrix)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isscalar(t) && isfinite(t) && t >= 0);" } },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("timeit"), slug: Some("timeit"),
    summary: "Measure function-handle execution time by repeated evaluation.",
    description: "`t = timeit(f)` repeatedly evaluates zero-argument function handle `f` and returns the median execution time in seconds. The optional output count controls how many outputs are requested from each invocation.",
    keywords: &["timeit", "benchmark", "timing", "performance", "gpu"],
    related: &["tic", "toc", "pause"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "The callback is warmed once, then evaluated in calibrated batches. At least seven batches contribute per-invocation samples; the median reduces outlier sensitivity.",
        "Callback results are discarded. Validate their contents inside the handle if required.",
        "The invocation's monotonic timing service measures the callback. `timeit` itself has no GPU kernel or fusion operation; work inside the handle keeps its existing provider and residency behavior. For asynchronous GPU work, synchronize inside the handle when that work must be included.",
        "A callback error propagates immediately. Cancellation abandons the pending callback and does not replay an already completed invocation.",
        "A typed-integer `numOutputs` and explicitly GPU-resident output count are separately gated RunMat extensions; ordinary nonnegative integer-valued double input is portable.",
    ] }],
    examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does `timeit` return?", answer: "A scalar double containing the median runtime per invocation in seconds." },
        BuiltinDocumentationFaq { question: "How many runs does `timeit` perform?", answer: "It selects a loop count so batches last a few milliseconds, then collects at least seven batches." },
        BuiltinDocumentationFaq { question: "Does `timeit` synchronize GPU kernels?", answer: "No. Synchronize inside the handle when the timed operation needs it." },
        BuiltinDocumentationFaq { question: "Can I time functions that require inputs?", answer: "Yes. Capture them in the handle, for example `timeit(@() myfun(A, B))`." },
        BuiltinDocumentationFaq { question: "How do I time multiple outputs?", answer: "Pass `timeit(@() svd(A), 3)`. RunMat requests three callback outputs and discards them. Typed-integer counts require RunMat compatibility mode." },
        BuiltinDocumentationFaq { question: "Why do successive runs differ?", answer: "System jitter, cache effects, and GPU scheduling vary; the median mitigates outliers." },
        BuiltinDocumentationFaq { question: "Can `timeit` time scripts?", answer: "Wrap the script body in a zero-argument function handle." },
        BuiltinDocumentationFaq { question: "Does `timeit` participate in fusion or JIT tiers?", answer: "It executes the provided handle; tiering or fusion happens inside that function." },
        BuiltinDocumentationFaq { question: "What happens if the function errors?", answer: "The error propagates immediately and timing stops." },
        BuiltinDocumentationFaq { question: "Is there a limit on runs?", answer: "Yes. The inner loop is capped at about one million iterations." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tic", target: BuiltinDocumentationLinkTarget::Builtin("tic") },
        BuiltinDocumentationLink { label: "toc", target: BuiltinDocumentationLinkTarget::Builtin("toc") },
        BuiltinDocumentationLink { label: "pause", target: BuiltinDocumentationLinkTarget::Builtin("pause") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "timeit runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/timing/timeit.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "timeit callback and timing tests", location: "crates/runmat-runtime/src/builtins/timing/timeit.rs tests" }],
        notes: &["Browser and real-provider examples require integrated execution gates; source catalog declarations alone are not execution evidence."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

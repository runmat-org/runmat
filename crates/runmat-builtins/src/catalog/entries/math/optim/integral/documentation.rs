use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("integral"),
    slug: Some("integral"),
    summary: "Approximate finite scalar definite integrals in MATLAB and RunMat.",
    description: "`q = integral(fun, a, b)` approximates a definite integral using adaptive quadrature. The current RunMat implementation accepts finite real scalar bounds and scalar-valued real callbacks; array-valued callbacks and other broader public forms remain an explicit implementation gap.",
    keywords: &["integral", "numerical integration", "adaptive quadrature", "quadrature", "function handle"],
    related: &["trapz", "cumtrapz", "fzero"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The callback can be a named function handle, anonymous function, or function-handle string. By default it receives row-vector batches of sample locations and must return a real row vector of the same shape. The adaptive solver batches its initial three samples and each subsequent pair.",
            "`Vectorized=false` selects scalar callback calls and is available in the current MATLAB compatibility mode. Array-valued and complex-valued callbacks, complex contours, waypoints, and infinite bounds are not yet implemented. `ArrayValued=true` is rejected rather than misinterpreted as a scalar call.",
            "Finite reversed bounds negate the result; equal bounds return zero without invoking the callback.",
            "`AbsTol`, `RelTol`, and exact structural `MaxFunEvals`/`MaxIntervalCount` controls are accepted as name/value pairs or scalar options-struct fields. `ArrayValued=false` is admitted. The newer `AbsoluteTolerance` and `RelativeTolerance` spellings still need separate compatibility-pinned implementation.",
            "Typed-integer bounds and numeric options are separately gated RunMat extensions; floating conversions require exact binary64 representation. Counts remain structural integers.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The adaptive solver runs on the host and invokes user code. A callback may itself call GPU-aware builtins, but `integral` does not provide a GPU quadrature kernel or fuse callback calls.",
            "Resident operands use the existing owning gather boundary; callback effects, cancellation, and source errors must be preserved through the runtime call path.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Integrate a polynomial anonymous function", program: "q = integral(@(x) x.^2, 0, 1)", display_output: Some("q =\n    0.3333"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(q-1/3)<1e-5);" } },
        BuiltinExample { id: "example-2", title: "Integrate sine over one half-period", program: "q = integral(@sin, 0, pi)", display_output: Some("q =\n    2.0000"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(q-2)<1e-5);" } },
        BuiltinExample { id: "example-3", title: "Set tighter tolerances", program: "q = integral(@sin, 0, pi, 'AbsTol', 1e-12, 'RelTol', 1e-8)", display_output: Some("q =\n    2.0000"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(q-2)<1e-7);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Is this the same as trapz?", answer: "No. `integral` adaptively evaluates a function over continuous bounds; `trapz` integrates sampled data." },
        BuiltinDocumentationFaq { question: "Does integral support vector-valued functions?", answer: "The default scalar-valued callback must accept a vector of sample locations and return a matching vector. A callback that returns a vector for each scalar location (`ArrayValued=true`) is not yet supported." },
        BuiltinDocumentationFaq { question: "Does integral run on the GPU?", answer: "The adaptive solver runs on the host. Its callback may call GPU-aware builtins." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "trapz", target: BuiltinDocumentationLinkTarget::Builtin("trapz") },
        BuiltinDocumentationLink { label: "cumtrapz", target: BuiltinDocumentationLinkTarget::Builtin("cumtrapz") },
        BuiltinDocumentationLink { label: "fzero", target: BuiltinDocumentationLinkTarget::Builtin("fzero") },
        BuiltinDocumentationLink { label: "MATLAB integral reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/integral.html") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/integral.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[
        BuiltinDocumentationLink { label: "Integral runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/integral.rs") },
        BuiltinDocumentationLink { label: "Callback invocation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common.rs") },
        BuiltinDocumentationLink { label: "Adaptive Simpson traversal", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/quadrature/adaptive.rs") },
    ], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar integrals and options", location: "builtins::math::optim::integral::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Anonymous callback", location: "runmat-vm/tests/closures.rs::integral_accepts_anonymous_function" },
    ], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

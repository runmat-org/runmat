use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("integral"),
    slug: Some("integral"),
    summary: "Approximate finite scalar definite integrals in MATLAB and RunMat.",
    description: "`q = integral(fun, a, b)` approximates a definite integral using adaptive quadrature. The current RunMat implementation accepts finite real scalar bounds and finite real scalar-valued callbacks; broader documented MATLAB forms remain an explicit implementation gap.",
    keywords: &["integral", "numerical integration", "adaptive quadrature", "quadrature", "function handle"],
    related: &["trapz", "cumtrapz", "fzero"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The callback can be a named function handle, anonymous function, or function-handle string. The current solver evaluates scalar sample points and uses adaptive Simpson error estimates.",
            "The callback must return a finite real scalar. Array-valued and complex-valued callbacks, complex contours, waypoints, infinite bounds, and vectorized batch callback execution are not yet implemented. `ArrayValued=true` is rejected rather than misinterpreted as a scalar call.",
            "Finite reversed bounds negate the result; equal bounds return zero without invoking the callback.",
            "`AbsTol`, `RelTol`, and exact structural `MaxFunEvals`/`MaxIntervalCount` controls are accepted as name/value pairs or scalar options-struct fields. `ArrayValued=false` is admitted. The newer `AbsoluteTolerance`, `RelativeTolerance`, and `Vectorized` spellings need separate compatibility-pinned implementation and are not silently claimed here.",
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
        BuiltinDocumentationFaq { question: "Does integral support vector-valued functions?", answer: "Not yet. The current implementation supports finite real scalar-valued callbacks." },
        BuiltinDocumentationFaq { question: "Does integral run on the GPU?", answer: "The adaptive solver runs on the host. Its callback may call GPU-aware builtins." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "trapz", target: BuiltinDocumentationLinkTarget::Builtin("trapz") },
        BuiltinDocumentationLink { label: "cumtrapz", target: BuiltinDocumentationLinkTarget::Builtin("cumtrapz") },
        BuiltinDocumentationLink { label: "fzero", target: BuiltinDocumentationLinkTarget::Builtin("fzero") },
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

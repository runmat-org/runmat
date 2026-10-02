use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("integral"),
    slug: Some("integral"),
    summary: "Approximate real definite and improper integrals with adaptive quadrature.",
    description: "`q = integral(fun, a, b)` approximates a real integral using an interior-sampling adaptive Gauss–Kronrod rule. Finite and infinite real bounds, endpoint singularities with integrable interior values, and real scalar or fixed-shape array-valued callbacks are supported. Monotone finite real `Waypoints` define an initial mesh. Complex contours and complex callback values remain implementation gaps.",
    keywords: &["integral", "numerical integration", "adaptive quadrature", "quadrature", "function handle"],
    related: &["trapz", "cumtrapz", "fzero"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The callback can be a named function handle, anonymous function, or function-handle string. By default it receives row-vector batches of interior sample locations and must return a real row vector of the same shape. The current rule begins with 15 samples per initial subinterval and checks `MaxFunEvals` before invoking any batch.",
            "`Vectorized=false` selects scalar callback calls and is available in the current MATLAB compatibility mode. With `ArrayValued=true`, each scalar sample must return a finite real array of a fixed shape; the result has that shape. `Vectorized` does not change this array-valued callback form. Complex-valued callbacks, complex contours, and unordered real waypoints are not yet implemented.",
            "`Waypoints` accepts a monotone finite real vector within the bounds. The points form an initial integration mesh; finite spans and infinite tails share one global error test, function-evaluation limit, interval limit, and invocation cancellation budget. This is the currently implemented RunMat form, not a claim of complete MATLAB waypoint parity.",
            "Finite reversed bounds negate the result. Equal bounds return zero without a callback in scalar mode; array-valued mode evaluates once to determine the result shape and returns an array of zeros.",
            "`AbsTol` and `RelTol` set the global acceptance tolerance. `MaxFunEvals` limits callback sample evaluations. RunMat's `MaxIntervalCount` extension separately limits adaptive subintervals; it is a positive structural integer and is not a synonym for `MaxFunEvals` or a claim of MATLAB `integral` option parity. These controls accept name/value pairs or scalar options-struct fields. The newer `AbsoluteTolerance` and `RelativeTolerance` spellings still need separate compatibility-pinned implementation.",
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
        BuiltinExample { id: "example-4", title: "Integrate a fixed-shape array-valued function", program: "q = integral(@(x) [x, x.^2], 0, 1, 'ArrayValued', true)", display_output: Some("q =\n    0.5000    0.3333"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(q),[1 2])); assert(abs(q(1)-1/2)<1e-5); assert(abs(q(2)-1/3)<1e-5);" } },
        BuiltinExample { id: "example-5", title: "Use real waypoints as an initial integration mesh", program: "q = integral(@(x) x.^2, 0, 1, 'Waypoints', [0.25 0.75])", display_output: Some("q =\n    0.3333"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(q-1/3)<1e-5);" } },
        BuiltinExample { id: "example-6", title: "Integrate an endpoint singularity without sampling the endpoint", program: "q = integral(@(x) 1./sqrt(x), 0, 1)", display_output: Some("q =\n    2.0000"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(q-2)<1e-5);" } },
        BuiltinExample { id: "example-7", title: "Integrate a decaying infinite tail", program: "q = integral(@(x) exp(-x), 0, Inf)", display_output: Some("q =\n    1.0000"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(q-1)<1e-6);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Is this the same as trapz?", answer: "No. `integral` adaptively evaluates a function over continuous bounds; `trapz` integrates sampled data." },
        BuiltinDocumentationFaq { question: "Does integral support vector-valued functions?", answer: "Yes. With `ArrayValued=true`, the callback receives one scalar location and returns a fixed-shape real numeric array. Otherwise, the default callback receives a vector of locations and returns matching scalar values." },
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
        BuiltinDocumentationLink { label: "Open Gauss–Kronrod rule", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/quadrature/open_rule.rs") },
        BuiltinDocumentationLink { label: "Global adaptive mesh", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/quadrature/mesh.rs") },
        BuiltinDocumentationLink { label: "Array-valued callback owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/integral/array.rs") },
        BuiltinDocumentationLink { label: "Waypoint geometry and shared segment budget", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/integral/path.rs") },
    ], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar integrals and options", location: "builtins::math::optim::integral::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Anonymous callback", location: "runmat-vm/tests/closures.rs::integral_accepts_anonymous_function" },
    ], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

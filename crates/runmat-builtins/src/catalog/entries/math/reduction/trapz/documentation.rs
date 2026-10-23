use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("trapz"), slug: Some("trapz"),
    summary: "Approximate integrals of sampled data using the trapezoidal rule with MATLAB-compatible dimension handling.",
    description: concat!(
            "`trapz(Y)` approximates integrals of single- or double-precision sampled data using trapezoidal segments. By default integration runs along the first ",
            "non-singleton dimension, with MATLAB-compatible explicit-dimension forms supported. Typed-integer sample data and spacing are available in RunMat mode",
            " when every value can enter the binary64 integration domain exactly.",
        ),
    keywords: &["trapz", "trapezoidal integration", "numerical integration", "discrete integration", "gpu"],
    related: &["cumtrapz", "diff", "sum", "cumsum", "gpuArray", "gather"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`trapz(Y)` assumes unit spacing between adjacent samples.",
            "`trapz(X, Y)` accepts a scalar spacing, a vector of coordinates whose length matches the working dimension, or an array `X` with the same size as `Y`.",
            "`trapz(..., dim)` selects the working dimension explicitly. Dimensions larger than `ndims(Y)` behave like singleton axes, so the result is zero along those slices.",
            "Logical inputs are promoted to double precision before integration. Complex-valued `Y` inputs are integrated component-wise and preserve their complex output type.",
            "Single-precision sample data produces single-precision output; typed-integer sample data is a RunMat extension and produces double output.",
            concat!(
            "Native typed-integer `Y` and `X` roles are independently compatibility-gated and require exact binary64 conversion. Native typed-integer `dim` is an o",
            "rdinary exact structural selector.",
        ),
            "When the selected dimension has length 0 or 1, `trapz` returns zeros because there are no intervals to integrate.",
        ]},
        BuiltinDocumentationSection { heading: "Forms, classes, and shapes", paragraphs: &[
            concat!(
            "The two-argument parser selects Y/dim when the first input is nonscalar and the second is a scalar dimension candidate or accepted empty control; a sc",
            "alar first input selects X/Y. Three inputs select X/Y/dim. Real scalar or empty spacing uses a constant or unit width; vector spacing supplies coordin",
            "ate differences, and same-size X supplies per-lane coordinate differences. Nonfinite spacing propagates through arithmetic rather than being silently ",
            "replaced.",
        ),
            concat!(
            "Selected zero/singleton axes return zeros and become extent one. Axes beyond the stored rank are virtual singleton axes and return zeros in the stored",
            " shape without allocating trailing-axis metadata. Single samples return single; other admitted real samples return double. Floating complex samples pr",
            "eserve their domain and single/double class. Host trapezoidal arithmetic uses double component accumulators before single publication.",
        ),
        ]},
        BuiltinDocumentationSection { heading: "Execution and allocation", paragraphs: &[
            concat!(
            "Inputs and controls retain their invocation owners across suspension. Each call uses its selected execution context and cancellation authority. Provid",
            "er outputs are checked for owner, device, physical class, shape, and forbidden input aliasing. Operational errors, cancellation, authority loss, and i",
            "nvalid attempted outputs remain terminal; no host replay follows them.",
        ),
            concat!(
            "Eligible resident forms stay with the input's exact provider. Automatic resident placement may select an admitted host-and-restore route only before a",
            " provider attempt. Explicit gpuArray data and resident controls that require host staging are rejected before gathering; allocation policy alone is no",
            "t placement consent. A later no-attempt in a multi-stage operation is terminal after earlier provider effects.",
        ),
            concat!(
            "Finite host allocation is not yet implemented for this operation's array scratch, shape, and output collections. Such paths reject a Managed allocatio",
            "n policy before unowned work; ordinary Unbounded host execution remains available. This restriction is an unfinished allocation capability, not a prom",
            "ise of a bounded host implementation. Native/browser execution shares the Runtime binding; long synchronous host loops do not promise browser yielding",
            ".",
        ),
            concat!(
            "These operations execute through their Runtime bindings, not a certified fused reduction. The live planner declines fused prod and nnz execution. Dist",
            "ributed execution is not declared by these entries.",
        ),
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Integrating sampled sine data with explicit coordinates",
            program: "x = 0:0.01:pi;\ny = sin(x);\narea = trapz(x, y)",
            display_output: Some("area ≈ 2"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(area-2)<0.0001);" },
        },
        BuiltinExample {
            id: "example-2", title: "Using unit spacing for a row vector",
            program: "y = [1 2 3];\nq = trapz(y)",
            display_output: Some("q = 4"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(q==4);" },
        },
        BuiltinExample {
            id: "example-3", title: "Integrating each row independently",
            program: "A = [1 2 3; 4 5 6];\nq = trapz(A, 2)",
            display_output: Some("q =\n     4\n    10"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(q,[4;10]));" },
        },
        BuiltinExample {
            id: "example-4", title: "Running trapz on GPU data",
            program: "G = gpuArray([1 2 3]);\nq = trapz(G);\nresult = gather(q)",
            display_output: Some("result = 4"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isgpuarray(q)); assert(result==4);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What spacing does `trapz(Y)` assume?", answer: "It assumes adjacent samples are one unit apart." },
        BuiltinDocumentationFaq { question: "What forms of `X` are accepted?", answer: "A scalar spacing, a vector of coordinates matching the integration dimension, or an array with the same size as `Y`." },
        BuiltinDocumentationFaq { question: "What happens for a scalar or singleton slice?", answer: "The integral is zero because there are no trapezoids to sum." },
        BuiltinDocumentationFaq { question: "Does trapz support complex inputs?", answer: "Yes. RunMat integrates the real and imaginary parts independently and returns a complex result." },
        BuiltinDocumentationFaq { question: "Does trapz keep gpuArray residency?", answer: concat!(
            "Eligible floating real or complex sample data can remain resident through the owning provider's trapz_dim operation. Logical and integer samples requi",
            "re admitted host conversion. Unsupported explicit gpuArray data or controls are rejected before input gathering; automatic placement may use admitted ",
            "fallback after no attempt.",
        ) },
    ],
    links: &[
        BuiltinDocumentationLink { label: "cumtrapz", target: BuiltinDocumentationLinkTarget::Builtin("cumtrapz") },
        BuiltinDocumentationLink { label: "diff", target: BuiltinDocumentationLinkTarget::Builtin("diff") },
        BuiltinDocumentationLink { label: "sum", target: BuiltinDocumentationLinkTarget::Builtin("sum") },
        BuiltinDocumentationLink { label: "cumsum", target: BuiltinDocumentationLinkTarget::Builtin("cumsum") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/math/reduction/trapz.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/trapz.rs") },
        BuiltinDocumentationLink { label: "MATLAB reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/trapz.html") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/math/reduction/trapz.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/trapz.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::math::reduction::trapz::tests", location: "builtins::math::reduction::trapz::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

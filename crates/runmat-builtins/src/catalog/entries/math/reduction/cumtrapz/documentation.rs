use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cumtrapz"), slug: Some("cumtrapz"),
    summary: "Compute cumulative trapezoidal integration in MATLAB and RunMat.",
    description: concat!(
            "`cumtrapz(Y)` computes cumulative trapezoidal integrals of sampled data in `Y`. The output matches `Y` in size, with zero at the first element of the ",
            "integration dimension and running integral values thereafter.",
        ),
    keywords: &["cumtrapz", "cumulative trapezoidal integration", "numerical integration", "discrete integration", "gpu"],
    related: &["trapz", "cumsum", "diff", "gpuArray", "gather"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`cumtrapz(Y)` assumes unit spacing between adjacent samples.",
            concat!(
            "`cumtrapz(X, Y)` accepts documented single- or double-precision scalar spacing or a coordinate vector whose length matches the working dimension. A sa",
            "me-size coordinate tensor is an explicitly gated RunMat extension.",
        ),
            "`cumtrapz(..., dim)` selects the working dimension explicitly. Dimensions larger than `ndims(Y)` behave like singleton axes, so the result stays zero along those slices.",
            concat!(
            "Documented `Y` classes are single and double, real or complex, and output preserves single versus double. Typed-integer and logical `Y`, typed-integer",
            " or logical `X`, and typed-integer `dim` are separate RunMat-only extensions.",
        ),
            concat!(
            "Typed-integer `Y` and `X` cross the binary64 integration boundary only when every authoritative integer value is exactly representable; otherwise RunM",
            "at reports an error instead of silently rounding.",
        ),
            "When the selected dimension has length 0 or 1, the output is zero along that dimension because there are no intervals to accumulate.",
        ]},
        BuiltinDocumentationSection { heading: "Forms, classes, and shapes", paragraphs: &[
            concat!(
            "The two-argument parser selects Y/dim for nonscalar first input and scalar dimension candidate or accepted empty control; scalar first input instead s",
            "elects X/Y. Three inputs select X/Y/dim. Empty spacing selects unit width. Scalar, vector, and gated same-size tensor coordinates preserve signed widt",
            "hs, including descending coordinates; nonfinite values propagate through arithmetic.",
        ),
            concat!(
            "Every output keeps Y's full shape and starts at zero along the selected axis. Zero/singleton and virtual trailing axes contain zeros. Axes beyond the ",
            "rank do not allocate dimension-sized metadata. Native single/double class and floating complex domain follow Y, not X; host arithmetic uses double com",
            "ponent accumulators before matching single publication. Sparse and complex-integer samples are unsupported.",
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
            id: "example-1", title: "Computing a cumulative integral with explicit coordinates",
            program: "x = [0 1 3];\ny = [0 1 2];\nq = cumtrapz(x, y)",
            display_output: Some("q = [0 0.5 3.5]"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(q,[0 0.5 3.5]));" },
        },
        BuiltinExample {
            id: "example-2", title: "Using unit spacing for sampled data",
            program: "y = [1 2 3];\nq = cumtrapz(y)",
            display_output: Some("q = [0 1.5 4]"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(q,[0 1.5 4]));" },
        },
        BuiltinExample {
            id: "example-3", title: "Accumulating across rows in a matrix",
            program: "A = [1 2 3; 4 5 6];\nq = cumtrapz(A, 2)",
            display_output: Some("q =\n     0    1.5    4\n     0    4.5   10"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(q,[0 1.5 4;0 4.5 10]));" },
        },
        BuiltinExample {
            id: "example-4", title: "Running cumtrapz on GPU data",
            program: "G = gpuArray([1 2 3]);\nq = cumtrapz(G);\nresult = gather(q)",
            display_output: Some("result = [0 1.5 4]"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isgpuarray(q)); assert(isequal(result,[0 1.5 4]));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How is `cumtrapz` different from `trapz`?", answer: "`trapz` returns the final integrated value, while `cumtrapz` returns the running integral at every sample." },
        BuiltinDocumentationFaq { question: "Why is the first element zero?", answer: "There is no interval before the first sample, so the cumulative integral begins at zero." },
        BuiltinDocumentationFaq { question: "What forms of `X` are accepted?", answer: concat!(
            "The compatibility surface is a single- or double-precision scalar spacing or coordinate vector matching the integration dimension. RunMat mode additio",
            "nally admits same-size coordinate tensors behind the `cumtrapz-tensor-spacing` extension gate.",
        ) },
        BuiltinDocumentationFaq { question: "Does cumtrapz support complex inputs?", answer: "Yes. RunMat accumulates the real and imaginary components independently and returns a complex result." },
        BuiltinDocumentationFaq { question: "Does cumtrapz keep gpuArray residency?", answer: concat!(
            "Eligible floating real or complex samples remain resident when the owning provider returns the required shape, class, precision, device, and fresh all",
            "ocation. Automatic placement may use an admitted host route before a provider attempt. Explicit resident data or controls cannot silently gather; exec",
            "ution failures and malformed outputs are terminal.",
        ) },
    ],
    links: &[
        BuiltinDocumentationLink { label: "trapz", target: BuiltinDocumentationLinkTarget::Builtin("trapz") },
        BuiltinDocumentationLink { label: "cumsum", target: BuiltinDocumentationLinkTarget::Builtin("cumsum") },
        BuiltinDocumentationLink { label: "diff", target: BuiltinDocumentationLinkTarget::Builtin("diff") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/math/reduction/cumtrapz.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/cumtrapz.rs") },
        BuiltinDocumentationLink { label: "MATLAB reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/cumtrapz.html") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/math/reduction/cumtrapz.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/cumtrapz.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::math::reduction::cumtrapz::tests", location: "builtins::math::reduction::cumtrapz::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

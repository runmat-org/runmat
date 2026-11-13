use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gradient"), slug: Some("gradient"),
    summary: "Compute numerical gradients in MATLAB and RunMat.",
    description: concat!(
            "`gradient(F)` computes numerical derivatives using central differences in the interior and one-sided differences at boundaries. Default dimension beha",
            "vior and multi-output ordering follow MATLAB semantics.",
        ),
    keywords: &["gradient", "numerical gradient", "finite difference", "derivative", "gpu"],
    related: &["diff", "meshgrid", "quiver", "gpuArray", "gather"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`gradient(F)` chooses the first non-singleton dimension for vectors and returns the column-direction derivative for matrices.",
            "`[FX, FY] = gradient(F)` on a matrix returns `FX` for dimension 2 (across columns) and `FY` for dimension 1 (down rows), matching MATLAB.",
            "`gradient(F, h)` applies the same scalar spacing to every returned dimension. For vector inputs, `h` may be a coordinate vector with one point per element.",
            "`gradient(F, hx, hy, ...)` accepts one scalar or coordinate-vector spacing per output dimension, ordered as MATLAB returns outputs (`x`/columns first for matrices, then `y`/rows).",
            "Coordinate-vector spacings must be finite, match the corresponding dimension length, and avoid zero finite-difference denominators.",
            "With scalar spacing, interior points use central differences `(f(i+1) - f(i-1)) / (2*h)`, while the first and last points use one-sided differences.",
            "With coordinate-vector spacing, denominators come from the supplied coordinates: boundary points use adjacent coordinate differences and interior points use `x(i+1) - x(i-1)`.",
            concat!(
            "Real, logical, and scalar numeric inputs promote through the standard tensor conversion path. Complex host inputs are supported by differentiating the",
            " real and imaginary parts independently.",
        ),
            concat!(
            "Eligible real or complex floating resident samples use gradient_dim for unit or scalar spacing. Coordinate spacing currently selects host evaluation; ",
            "the public Runtime does not call gradient_dim_with_coordinates. Explicit resident data or controls requiring host fallback are rejected before gatheri",
            "ng.",
        ),
        ]},
        BuiltinDocumentationSection { heading: "Forms, classes, and shapes", paragraphs: &[
            concat!(
            "Each result keeps the sample shape; rank-one storage is exposed as a MATLAB-style row vector. A vector or scalar has one gradient output; a nonvector ",
            "matrix has two. Higher-rank arrays expose one output per axis, ordered 2,1,3,... . Requests beyond the available components fail before execution. Sin",
            "gleton axes produce zeros and empty shapes remain empty.",
        ),
            concat!(
            "Single samples return single; other admitted real samples return double. Floating complex samples return matching complex precision. Host stencil arit",
            "hmetic uses double component accumulators before single publication. Integer/logical data and spacing are separately gated RunMat extensions; integer ",
            "values must enter binary64 exactly. Sparse and complex-integer samples are unsupported.",
        ),
            concat!(
            "A shared scalar spacing must be finite and nonzero. Coordinate data is read in column-major order, must match its axis length, be finite, and have non",
            "zero adjacent and central denominators. For array gradients, coordinate spacing requires one argument per gradient dimension; no missing spacing is in",
            "vented.",
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
            "t placement consent. A later no-attempt in a multi-stage operation is terminal after earlier provider effects. The Runtime currently calls gradient_di",
            "m only for unit/scalar spacing; provider gradient_dim_with_coordinates exists but is not used by this public binding.",
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
            id: "example-1", title: "Differentiating a row vector",
            program: "v = [1 4 9];\ng = gradient(v)",
            display_output: Some("g = [3 4 5]"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(g,[3 4 5]));" },
        },
        BuiltinExample {
            id: "example-2", title: "Requesting both matrix gradient components",
            program: "A = [1 2; 3 4];\n[FX, FY] = gradient(A)",
            display_output: Some("FX =\n     1     1\n     1     1\n\nFY =\n     2     2\n     2     2"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(FX,ones(2))); assert(isequal(FY,2*ones(2)));" },
        },
        BuiltinExample {
            id: "example-3", title: "Using scalar spacing on GPU data",
            program: "G = gpuArray(single([1 4 9]));\nD = gradient(G, 2);\nout = gather(D)",
            display_output: Some("out = single([1.5 2.0 2.5])"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isgpuarray(D)); assert(isa(out,'single')); assert(isequal(out,single([1.5 2 2.5])));" },
        },
        BuiltinExample {
            id: "example-4", title: "Using nonuniform coordinate spacing",
            program: "x = [0 1 3];\nv = [1 4 9];\ng = gradient(v, x)",
            display_output: Some("g = [3.0000 2.6667 2.5000]"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(g-[3 8/3 2.5]))<1e-12);" },
        },
        BuiltinExample {
            id: "example-5", title: "Feeding `gradient` into a vector-field plot",
            program: "[X, Y] = meshgrid(linspace(-2, 2, 25), linspace(-2, 2, 25));\nZ = X .* exp(-X.^2 - Y.^2);\n[DX, DY] = gradient(Z, X(1,2)-X(1,1), Y(2,1)-Y(1,1));\nquiver(X, Y, DX, DY)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements { runtime: &[BuiltinRuntimeCapability::PlotCore], ..BuiltinExampleRequirements::NONE },
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(size(DX),[25 25])); assert(isequal(size(DY),[25 25])); assert(abs(DX(13,13)-exp(-1/36))<1e-10); assert(abs(DY(13,13))<1e-12); assert(numel(get(gca,'Children'))>=1);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What finite-difference stencil does `gradient` use?", answer: "RunMat matches MATLAB's shape-preserving behavior: central differences in the interior and first-order one-sided differences at the boundaries." },
        BuiltinDocumentationFaq { question: "Why does a matrix return the x-direction first?", answer: "MATLAB defines the first matrix output along dimension 2 (columns), then dimension 1 (rows). RunMat preserves that ordering so plotting workflows like `quiver` line up correctly." },
        BuiltinDocumentationFaq { question: "Can I pass coordinate vectors for spacing?", answer: concat!(
            "Yes. Vector inputs support `gradient(F, X)`, and arrays support per-dimension scalar or coordinate-vector spacings such as `[FX, FY] = gradient(F, x, ",
            "y)`. Coordinate vectors must match the corresponding dimension length.",
        ) },
        BuiltinDocumentationFaq { question: "Does `gradient` support GPU arrays?", answer: concat!(
            "Eligible floating real or complex samples with default or scalar spacing can remain resident through gradient_dim. Coordinate spacing uses admitted ho",
            "st evaluation only for automatic placement. Explicit inputs requiring a missing resident route are rejected before gathering; provider failures do not",
            " trigger replay.",
        ) },
        BuiltinDocumentationFaq { question: "Do complex inputs work?", answer: concat!(
            "Yes. Host complex tensors are differentiated by real and imaginary parts independently, and providers with complex-interleaved gradient hook support k",
            "eep complex GPU tensors resident.",
        ) },
    ],
    links: &[
        BuiltinDocumentationLink { label: "diff", target: BuiltinDocumentationLinkTarget::Builtin("diff") },
        BuiltinDocumentationLink { label: "meshgrid", target: BuiltinDocumentationLinkTarget::Builtin("meshgrid") },
        BuiltinDocumentationLink { label: "quiver", target: BuiltinDocumentationLinkTarget::Builtin("quiver") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
        BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
        BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/math/reduction/gradient.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/gradient.rs") },
        BuiltinDocumentationLink { label: "MATLAB reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/gradient.html") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/math/reduction/gradient.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/gradient.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::math::reduction::gradient::tests", location: "builtins::math::reduction::gradient::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "builtins::math::reduction::gradient::tests::gradient_gpu_scalar_spacing_matches_cpu_and_stays_resident", location: "builtins::math::reduction::gradient::tests::gradient_gpu_scalar_spacing_matches_cpu_and_stays_resident" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

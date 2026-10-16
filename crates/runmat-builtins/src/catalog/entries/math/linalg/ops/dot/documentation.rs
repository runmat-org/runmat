use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`dot(A, B)` evaluates inner products between matching slices of `A` and `B`, optionally along a selected dimension. Complex inputs use conjugate-on-first-argument semantics consistent with MATLAB and RunMat.",
            "Vectors need the same number of elements and may use different row/column orientations. Matrices and N-D arrays must have identical sizes.",
            "When no dimension is supplied, the function reduces along the first non-singleton dimension (`dim = 1` when all dimensions are singleton).",
            "`dot(A, B, dim)` collapses dimension `dim` (1-based) while leaving every other dimension untouched.",
            "Complex inputs conjugate the first argument before multiplication; real inputs use a straight element-wise product.",
            "Single inputs produce single results, including on the host/fallback path; otherwise the floating result is double.",
            "Empty reductions yield zeros of the appropriate shape; length mismatches raise `A and B must be the same size`.",
            "Public MATLAB data inputs are single or double. RunMat additionally accepts logical and all eight typed-integer classes only in RunMat extension mode.",
            "When both extension operands are typed integers, each product is formed from authoritative integer storage before crossing to the floating reduction/output domain, avoiding pre-product loss above flintmax.",
            "Typed-integer `dim` is a documented control input, is available in compatibility modes, and is decoded exactly in every supported integer class."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["Same-owner, same-device real floating inputs can use the owning provider's native dot hook. Other supported resident forms gather through the originating owner and restore real results there when its precision can represent the output. Complex outputs remain host-side until provider complex dot support is available.",
            "Native dispatch never sends integer or logical storage to a floating dot hook. It requires proven common ownership/device and accepts a fresh result only after owner, device, shape, real noninteger storage, and operand-requested precision validation. A malformed native result is rejected without replay; only a typed unsupported-hook result permits host fallback. Global-provider fallback for an unknown device is not accepted as ownership. Outputs that alias either input or the fallback residency anchor are rejected and never freed as provider-owned results. Other rejected allocations are freed only through a proven actual owner. Fallback gathers through each handle owner and restores real output to the first resident owner only after applying the same owner, device, shape, storage, class-metadata, freshness, and requested-precision checks.",
            "The provider-side hook is limited to proven-same-owner, same-device real floating inputs. Integer/logical extensions gather through the owning provider and bypass floating-native hooks; operational provider failures and malformed native outputs remain terminal."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "example-1", title: "Computing the dot product of row vectors", program: "A = [1 2 3];\nB = [4 5 6];\nval = dot(A, B)", display_output: Some("val = 32"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(val == 32);" } },
    BuiltinExample { id: "example-2", title: "Dotting column vectors to obtain a scalar", program: "u = [1; 3; 5];\nv = [2; 4; 6];\nval = dot(u, v)", display_output: Some("val = 44"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(val == 44);" } },
    BuiltinExample { id: "example-3", title: "Applying `dot` along a chosen dimension", program: "X = [1 2 3; 4 5 6];\nY = [6 5 4; 3 2 1];\ncols = dot(X, Y, 1)  % collapse rows\nrows = dot(X, Y, 2)  % collapse columns", display_output: Some("cols = [18 20 18];\nrows = [28; 28]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(cols, [18 20 18])); assert(isequal(rows, [28; 28]));" } },
    BuiltinExample { id: "example-4", title: "Dotting complex vectors uses conjugation on the first input", program: "a = [1+2i, 3-4i];\nb = [2-3i, -1+5i];\nval = dot(a, b)", display_output: Some("val = -27 + 4i"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(val == -27 + 4i);" } },
    BuiltinExample { id: "example-5", title: "Evaluating `dot` on `gpuArray` inputs", program: "G1 = gpuArray([1 2 3 4]);\nG2 = gpuArray([4 3 2 1]);\nG = dot(G1, G2);\nresult = gather(G)", display_output: Some("result = 20"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(result == 20);" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `dot` require vectors?", answer: "No. Any pair of tensors with identical sizes works; specifying a dimension lets you dot slices of higher-dimensional arrays." },
    BuiltinDocumentationFaq { question: "How does the optional dimension behave?", answer: "`dot(A, B, dim)` collapses the `dim`th dimension (1-based). Dimensions greater than the array rank have length 1 and therefore leave the data unchanged." },
    BuiltinDocumentationFaq { question: "What happens with complex numbers?", answer: "The first input is conjugated before multiplication so the result matches MATLAB's hermitian inner product." },
    BuiltinDocumentationFaq { question: "Are empty inputs supported?", answer: "Yes. If the reduction dimension has length 0 the result is filled with zeros of the appropriate shape." },
    BuiltinDocumentationFaq { question: "Will the result stay on the GPU?", answer: "When a provider is active the runtime uploads real-valued results back to the device. Complex outputs stay on the host until GPU complex support is available." },
    BuiltinDocumentationFaq { question: "What error is raised for size mismatches?", answer: "When `A` and `B` differ in size `dot` raises: `A and B must be the same size.` matching MATLAB's wording." },
    BuiltinDocumentationFaq { question: "Does `dot` accept logical or integer inputs?", answer: "Only as explicit RunMat extensions. Current public MATLAB documentation lists single and double data inputs. RunMat mode accepts logical and all eight integer classes; compatibility modes reject those data forms. Typed-integer `dim` remains documented and available." },
    BuiltinDocumentationFaq { question: "Can I request conjugation of the second argument instead?", answer: "No. MATLAB's `dot` is fixed to conjugate the first argument. Use `sum(A .* conj(B))` manually if you need the opposite orientation." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "mtimes", target: BuiltinDocumentationLinkTarget::Builtin("mtimes") },
    BuiltinDocumentationLink { label: "mldivide", target: BuiltinDocumentationLinkTarget::Builtin("mldivide") },
    BuiltinDocumentationLink { label: "norm", target: BuiltinDocumentationLinkTarget::Builtin("norm") },
    BuiltinDocumentationLink { label: "sum", target: BuiltinDocumentationLinkTarget::Builtin("sum") },
    BuiltinDocumentationLink { label: "ctranspose", target: BuiltinDocumentationLinkTarget::Builtin("ctranspose") },
    BuiltinDocumentationLink { label: "mpower", target: BuiltinDocumentationLinkTarget::Builtin("mpower") },
    BuiltinDocumentationLink { label: "mrdivide", target: BuiltinDocumentationLinkTarget::Builtin("mrdivide") },
    BuiltinDocumentationLink { label: "trace", target: BuiltinDocumentationLinkTarget::Builtin("trace") },
    BuiltinDocumentationLink { label: "transpose", target: BuiltinDocumentationLinkTarget::Builtin("transpose") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/dot.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dot"),
    slug: Some("dot"),
    summary: "Compute dot products in MATLAB and RunMat.",
    description: "`dot(A, B)` evaluates inner products between matching slices of `A` and `B`, optionally along a selected dimension. Complex inputs use conjugate-on-first-argument semantics consistent with MATLAB and RunMat.",
    keywords: &["dot", "inner product", "linear algebra", "gpu", "dimension"],
    related: &["mtimes", "mldivide", "norm", "sum", "ctranspose", "mpower", "mrdivide", "trace", "transpose"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/dot.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::ops::dot::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing runtime integration tests", location: "builtins::math::linalg::ops::dot::tests::dot_gpu_roundtrip" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

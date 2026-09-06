use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Element-wise multiplication and implicit expansion", paragraphs: &["`times(A, B)` and `A .* B` multiply corresponding elements. Singleton dimensions expand when the remaining extents are compatible; incompatible dimensions produce a size-mismatch error, and compatible empty dimensions remain empty.", "Real and complex operands may be combined. Character inputs contribute their Unicode code points and logical inputs contribute zero or one. String arrays are not numeric and are rejected."] },
    BuiltinDocumentationSection { heading: "Numeric classes and sparse storage", paragraphs: &["Double inputs produce double. An operation involving single floating-point data produces single. Fixed-width integer multiplication accepts matching integer classes or one integer operand with scalar double, preserves the integer class, rounds according to the integer arithmetic contract, and saturates at the class bounds.", "Sparse-sparse, sparse-scalar, and compatible dense products retain real sparse storage when the mathematical result remains sparse. Complex products use full complex storage because complex sparse storage is not currently available."] },
    BuiltinDocumentationSection { heading: "Accelerated execution and output prototypes", paragraphs: &["Matching provider-resident operands use element-wise multiplication when available; scalar multiplication and provider-side expansion keep supported work resident. Unsupported forms gather through the owning provider and use the same host rules.", "RunMat mode accepts `times(A, B, 'like', prototype)`. The prototype controls host or provider residency and may request complex output. MATLAB compatibility mode rejects this extension."] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "matrices", title: "Multiply two matrices element by element", program: "A = [1 2 3; 4 5 6];\nB = [7 8 9; 1 2 3];\nP = times(A, B)", display_output: Some("P = [7 16 27; 4 10 18]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(P, [7 16 27; 4 10 18]));" } },
    BuiltinExample { id: "scalar", title: "Scale an array", program: "A = [8 1 6; 3 5 7; 4 9 2];\nscaled = times(A, 0.5)", display_output: Some("scaled = [4 0.5 3; 1.5 2.5 3.5; 2 4.5 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(scaled, A .* 0.5));" } },
    BuiltinExample { id: "implicit-expansion", title: "Expand a column and row", program: "col = (1:3)';\nrow = [10 20 30];\nP = times(col, row)", display_output: Some("P = [10 20 30; 20 40 60; 30 60 90]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(P, [10 20 30; 20 40 60; 30 60 90]));" } },
    BuiltinExample { id: "complex", title: "Multiply complex values", program: "z1 = [1+2i, 3-4i];\nz2 = [2-1i, -1+1i];\nP = times(z1, z2)", display_output: Some("P = [4+3i 1+7i]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(P, [4+3i 1+7i]));" } },
    BuiltinExample { id: "characters", title: "Multiply character code points", program: "codes = times('ABC', 2)", display_output: Some("codes = [130 132 134]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(codes, [130 132 134]));" } },
    BuiltinExample { id: "gpu", title: "Multiply provider-resident arrays", program: "G1 = gpuArray(single([1 2 3]));\nG2 = gpuArray(single([4 5 6]));\ndeviceProduct = times(G1, G2);\nP = gather(deviceProduct)", display_output: Some("P = single([4 10 18])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(deviceProduct, 'gpuArray')); assert(isequal(P, single([4 10 18])));" } },
    BuiltinExample { id: "gpu-like", title: "Request a provider-resident product", program: "A = [1 2 3];\nB = [4 5 6];\nprototype = gpuArray(0);\ndeviceProduct = times(A, B, 'like', prototype);\nP = gather(deviceProduct)", display_output: Some("P = [4 10 18]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(deviceProduct, 'gpuArray')); assert(isequal(P, [4 10 18]));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does times use implicit expansion?", answer: "Yes. Singleton dimensions expand when every non-singleton extent is compatible." },
    BuiltinDocumentationFaq { question: "What class does times return?", answer: "Floating arithmetic follows the participating floating class rules. Supported integer arithmetic preserves the integer class. Logical and character inputs produce numeric output." },
    BuiltinDocumentationFaq { question: "How does integer overflow behave?", answer: "Fixed-width integer products saturate at the minimum or maximum of their class." },
    BuiltinDocumentationFaq { question: "Can I multiply provider-resident arrays and host scalars?", answer: "Yes. RunMat keeps supported scalar multiplication on the provider and otherwise follows the owner-aware fallback path." },
    BuiltinDocumentationFaq { question: "How do I request a provider-resident result?", answer: "In RunMat mode, pass `'like'` and a provider-resident prototype." },
    BuiltinDocumentationFaq { question: "What happens with empty arrays?", answer: "Compatible empty dimensions propagate into the broadcasted output shape." },
    BuiltinDocumentationFaq { question: "Can times combine real and complex operands?", answer: "Yes. The result is complex and uses the same implicit-expansion rules." },
    BuiltinDocumentationFaq { question: "Does times preserve sparse matrices?", answer: "Real sparse storage is retained when the mathematical result remains sparse. Complex products currently use full complex storage." },
    BuiltinDocumentationFaq { question: "Does times accept string arrays?", answer: "No. String arrays are not numeric multiplication operands." },
];

const RELATED: &[&str] = &[
    "mtimes", "plus", "minus", "rdivide", "ldivide", "power", "gpuArray", "gather", "sparse",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "mtimes", target: BuiltinDocumentationLinkTarget::Builtin("mtimes") },
    BuiltinDocumentationLink { label: "plus", target: BuiltinDocumentationLinkTarget::Builtin("plus") },
    BuiltinDocumentationLink { label: "minus", target: BuiltinDocumentationLinkTarget::Builtin("minus") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic/times") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("times"),
    slug: Some("times"),
    summary: "Multiply arrays element by element.",
    description: "`times(A, B)` and `A .* B` multiply compatible numeric, logical, character, sparse, symbolic, or provider-resident operands using implicit expansion.",
    keywords: &[
        "times",
        "multiplication",
        "element-wise",
        "implicit expansion",
        "integer",
        "sparse",
        "gpu",
    ],
    related: RELATED,
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Element-wise multiplication runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic/times") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, dense, complex, sparse, integer, and provider behavior", location: "builtins::math::elementwise::binary_arithmetic::times::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Element-wise addition and implicit expansion",
        paragraphs: &[
            "`plus(A, B)` and `A + B` add corresponding elements. Equal dimensions align directly and singleton dimensions expand to the other operand. Incompatible non-singleton dimensions produce a size-mismatch error. Empty dimensions remain empty in the broadcasted result.",
            "Real and complex operands may be combined. Character inputs contribute their Unicode code points and logical inputs contribute zero or one. String-array append through `plus` is not implemented yet and remains an explicit compatibility gap.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Numeric classes and sparse storage",
        paragraphs: &[
            "Double inputs produce double. An operation involving single floating-point data produces single. Fixed-width integer addition accepts matching integer classes or one integer operand with scalar double, preserves the integer class, rounds fractional results according to the integer arithmetic contract, and saturates at the class bounds.",
            "Sparse-sparse addition retains sparse storage when the expanded result remains sparse. Operations whose implicit zeros become nonzero use full storage, subject to the runtime materialization limit. Complex and mixed sparse forms follow the supported sparse arithmetic paths or return a typed error.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated execution and output prototypes",
        paragraphs: &[
            "Compatible provider-resident operands use element-wise addition or scalar-add provider operations when available. Unsupported provider forms gather through the owning provider and use the same host arithmetic rules; provider failures are not reported as unsupported fallbacks.",
            "RunMat mode accepts `plus(A, B, 'like', prototype)`. A provider-resident prototype requests provider residency, a host prototype requests host residency, and a complex prototype promotes a real result to complex storage. This extension is rejected in MATLAB compatibility mode.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "matrices", title: "Add two matrices", program: "A = [1 2 3; 4 5 6];\nB = [7 8 9; 1 2 3];\nS = plus(A, B)", display_output: Some("S = [8 10 12; 5 7 9]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(S, [8 10 12; 5 7 9]));" } },
    BuiltinExample { id: "scalar", title: "Add a scalar to an array", program: "A = [8 1 6; 3 5 7; 4 9 2];\nshifted = plus(A, 0.5)", display_output: Some("shifted = [8.5 1.5 6.5; 3.5 5.5 7.5; 4.5 9.5 2.5]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(shifted, A + 0.5));" } },
    BuiltinExample { id: "implicit-expansion", title: "Expand a column and row", program: "col = (1:3)';\nrow = [10 20 30];\nS = plus(col, row)", display_output: Some("S = [11 21 31; 12 22 32; 13 23 33]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(S, [11 21 31; 12 22 32; 13 23 33]));" } },
    BuiltinExample { id: "complex", title: "Add complex values", program: "z1 = [1+2i, 3-4i];\nz2 = [2-1i, -1+1i];\nS = plus(z1, z2)", display_output: Some("S = [3+1i 2-3i]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(S, [3+1i 2-3i]));" } },
    BuiltinExample { id: "characters", title: "Add character code points", program: "codes = plus('ABC', 2)", display_output: Some("codes = [67 68 69]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(codes, [67 68 69]));" } },
    BuiltinExample { id: "integer-saturation", title: "Preserve a fixed-width integer class", program: "S = plus(uint8([250 255]), uint8([10 1]))", display_output: Some("S = uint8([255 255])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(S, 'uint8')); assert(isequal(S, uint8([255 255])));" } },
    BuiltinExample { id: "gpu-like", title: "Request a provider-resident result", program: "G1 = gpuArray([1 2 3]);\nG2 = gpuArray([4 5 6]);\nprototype = gpuArray(0);\ndeviceSum = plus(G1, G2, 'like', prototype);\nS = gather(deviceSum)", display_output: Some("S = [5 7 9]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(deviceSum, 'gpuArray')); assert(isequal(S, [5 7 9]));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does plus use implicit expansion?", answer: "Yes. Singleton dimensions expand to compatible dimensions of the other operand; incompatible non-singleton dimensions produce a size-mismatch error." },
    BuiltinDocumentationFaq { question: "What class does plus return?", answer: "Floating arithmetic follows the participating floating class rules. Integer arithmetic preserves the integer class when the operands use the same class or pair an integer with scalar double. Logical and character inputs produce numeric output." },
    BuiltinDocumentationFaq { question: "How does integer overflow behave?", answer: "Fixed-width integer sums saturate at the minimum or maximum of their class." },
    BuiltinDocumentationFaq { question: "Can host scalars be added to provider-resident arrays?", answer: "Yes. RunMat uses a provider scalar operation when supported and otherwise follows the owner-aware fallback path." },
    BuiltinDocumentationFaq { question: "How can the result be placed on a particular device?", answer: "In RunMat mode, pass `'like'` and a provider-resident prototype. A host prototype requests a host result." },
    BuiltinDocumentationFaq { question: "Does plus preserve sparse matrices?", answer: "It preserves sparse storage when the mathematical result remains sparse. Forms that turn implicit zeros into nonzero values require materialization." },
    BuiltinDocumentationFaq { question: "What happens with empty arrays?", answer: "Compatible empty dimensions propagate into the broadcasted output shape." },
    BuiltinDocumentationFaq { question: "Can plus append strings?", answer: "Not yet. String-array append through plus remains a known compatibility gap." },
];

const RELATED: &[&str] = &[
    "minus", "times", "rdivide", "ldivide", "power", "sum", "gpuArray", "gather", "sparse",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "minus", target: BuiltinDocumentationLinkTarget::Builtin("minus") },
    BuiltinDocumentationLink { label: "times", target: BuiltinDocumentationLinkTarget::Builtin("times") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic/plus") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("plus"),
    slug: Some("plus"),
    summary: "Add arrays element by element.",
    description: "`plus(A, B)` and `A + B` add compatible numeric, logical, character, sparse, symbolic, or provider-resident operands using implicit expansion.",
    keywords: &["plus", "addition", "element-wise", "implicit expansion", "integer", "sparse", "gpu"],
    related: RELATED,
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Element-wise addition runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic/plus") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, dense, complex, sparse, integer, and provider fallback behavior", location: "builtins::math::elementwise::binary_arithmetic::plus::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &["String-array append through plus is not implemented and is intentionally documented as a compatibility gap."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

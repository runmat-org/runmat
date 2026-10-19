use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Element-wise subtraction and implicit expansion", paragraphs: &["`minus(A, B)` and `A - B` subtract corresponding elements of `B` from `A`. Singleton dimensions expand when the remaining extents are compatible; incompatible dimensions produce a size-mismatch error, and compatible empty dimensions remain empty.", "Real and complex operands may be combined. Character inputs contribute their Unicode code points and logical inputs contribute zero or one. String arrays are not numeric and are rejected."] },
    BuiltinDocumentationSection { heading: "Numeric classes and sparse storage", paragraphs: &["Double inputs produce double. An operation involving single floating-point data produces single. Fixed-width integer subtraction accepts matching integer classes or one integer operand with scalar double, preserves the integer class, rounds according to the integer arithmetic contract, and saturates at the class bounds.", "Sparse-sparse subtraction retains sparse storage when the expanded result remains sparse. Dense, complex, and nonzero scalar forms use full storage when implicit zeros become nonzero, subject to the runtime materialization limit."] },
    BuiltinDocumentationSection { heading: "Accelerated execution and output prototypes", paragraphs: &["Matching provider-resident operands use element-wise subtraction when available. Provider paths distinguish `A - scalar` from `scalar - B`; implicit expansion may use provider `repmat`. Unsupported forms gather through the owning provider and use the same host rules.", "RunMat mode accepts `minus(A, B, 'like', prototype)`. The prototype controls host or provider residency and may request complex output. MATLAB compatibility mode rejects this extension."] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "matrices", title: "Subtract two matrices", program: "A = [7 8 9; 4 5 6];\nB = [1 2 3; 1 2 3];\nD = minus(A, B)", display_output: Some("D = [6 6 6; 3 3 3]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(D, [6 6 6; 3 3 3]));" } },
    BuiltinExample { id: "scalar", title: "Subtract a scalar from an array", program: "A = [8 1 6; 3 5 7; 4 9 2];\nshifted = minus(A, 0.5)", display_output: Some("shifted = [7.5 0.5 5.5; 2.5 4.5 6.5; 3.5 8.5 1.5]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(shifted, A - 0.5));" } },
    BuiltinExample { id: "implicit-expansion", title: "Expand a column and row", program: "col = (1:3)';\nrow = [10 20 30];\nD = minus(col, row)", display_output: Some("D = [-9 -19 -29; -8 -18 -28; -7 -17 -27]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(D, [-9 -19 -29; -8 -18 -28; -7 -17 -27]));" } },
    BuiltinExample { id: "complex", title: "Subtract complex values", program: "z1 = [1+2i, 3-4i];\nz2 = [2-1i, -1+1i];\nD = minus(z1, z2)", display_output: Some("D = [-1+3i 4-5i]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(D, [-1+3i 4-5i]));" } },
    BuiltinExample { id: "characters", title: "Subtract from character code points", program: "codes = minus('DEF', 1)", display_output: Some("codes = [67 68 69]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(codes, [67 68 69]));" } },
    BuiltinExample { id: "gpu-like", title: "Request a provider-resident difference", program: "G1 = gpuArray([10 20 30]);\nG2 = gpuArray([1 2 3]);\nprototype = gpuArray(0);\ndeviceDiff = minus(G1, G2, 'like', prototype);\nD = gather(deviceDiff)", display_output: Some("D = [9 18 27]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(deviceDiff, 'gpuArray')); assert(isequal(D, [9 18 27]));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does minus use implicit expansion?", answer: "Yes. Singleton dimensions expand when every non-singleton extent is compatible." },
    BuiltinDocumentationFaq { question: "What class does minus return?", answer: "Floating arithmetic follows the participating floating class rules. Supported integer arithmetic preserves the integer class. Logical and character inputs produce numeric output." },
    BuiltinDocumentationFaq { question: "How does integer overflow behave?", answer: "Fixed-width integer differences saturate at the minimum or maximum of their class." },
    BuiltinDocumentationFaq { question: "Can I subtract provider-resident arrays and host scalars?", answer: "Yes. The provider uses direction-aware scalar subtraction when supported and otherwise follows the owner-aware fallback path." },
    BuiltinDocumentationFaq { question: "How do I request a provider-resident result?", answer: "In RunMat mode, pass `'like'` and a provider-resident prototype." },
    BuiltinDocumentationFaq { question: "What happens with empty arrays?", answer: "Compatible empty dimensions propagate into the broadcasted output shape." },
    BuiltinDocumentationFaq { question: "Can minus combine real and complex operands?", answer: "Yes. The result is complex and uses the same implicit-expansion rules." },
    BuiltinDocumentationFaq { question: "Does minus preserve sparse matrices?", answer: "Sparse storage is retained when the mathematical result remains sparse; forms that make implicit zeros nonzero require full storage." },
    BuiltinDocumentationFaq { question: "Does minus accept string arrays?", answer: "No. String arrays are not numeric subtraction operands." },
];

const RELATED: &[&str] = &[
    "plus", "times", "rdivide", "ldivide", "power", "gpuArray", "gather", "sparse",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "plus", target: BuiltinDocumentationLinkTarget::Builtin("plus") },
    BuiltinDocumentationLink { label: "times", target: BuiltinDocumentationLinkTarget::Builtin("times") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic/minus") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("minus"),
    slug: Some("minus"),
    summary: "Subtract arrays element by element.",
    description: "`minus(A, B)` and `A - B` subtract compatible numeric, logical, character, sparse, symbolic, or provider-resident operands using implicit expansion.",
    keywords: &[
        "minus",
        "subtraction",
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
        implementation: &[BuiltinDocumentationLink { label: "Element-wise subtraction runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic/minus") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, dense, complex, sparse, integer, and provider behavior", location: "builtins::math::elementwise::binary_arithmetic::minus::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};
const SECTIONS: &[BuiltinDocumentationSection] = &[BuiltinDocumentationSection { heading: "Storage representation", paragraphs: &["`issparse(A)` returns one logical scalar that reports the storage representation, not the number of zero elements. Empty sparse matrices are sparse; dense matrices containing only zeros are not.", "Numeric, complex, logical, and RunMat extension integer sparse matrices return true. Scalars, dense arrays, text, containers, objects, callables, and current dense gpuArray handles return false."] }];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "sparse",
        title: "Check a sparse matrix",
        program: "S = sparse([1 3], [1 2], [4 -1], 3, 2);\ntf = issparse(S)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "dense",
        title: "Check a dense matrix",
        program: "A = [0 4; 0 0];\ntf = issparse(A)",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
    BuiltinExample {
        id: "full",
        title: "Observe full conversion",
        program: "S = sparse(3, 2);\nA = full(S);\ntf = [issparse(S), issparse(A)]",
        display_output: Some("tf = [true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0])));",
        },
    },
    BuiltinExample {
        id: "dense-gpu",
        title: "Check current dense gpuArray storage",
        program: "G = gpuArray(single([0 4; 0 0]));\ntf = issparse(G)",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `issparse` count zeros?",
        answer: "No. It checks the storage representation.",
    },
    BuiltinDocumentationFaq {
        question: "Is an empty sparse matrix sparse?",
        answer: "Yes. Shape and stored-element count do not change its sparse representation.",
    },
    BuiltinDocumentationFaq {
        question: "Are gpuArray values sparse?",
        answer: "RunMat's current gpuArray representation is dense, so they return false.",
    },
    BuiltinDocumentationFaq {
        question: "Does `issparse` return an array?",
        answer: "No. It returns one logical scalar.",
    },
];
const RELATED: &[&str] = &[
    "sparse",
    "full",
    "nnz",
    "isnumeric",
    "islogical",
    "isgpuarray",
];
const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink { label: "sparse", target: BuiltinDocumentationLinkTarget::Builtin("sparse") }, BuiltinDocumentationLink { label: "full", target: BuiltinDocumentationLinkTarget::Builtin("full") }, BuiltinDocumentationLink { label: "nnz", target: BuiltinDocumentationLinkTarget::Builtin("nnz") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/issparse.rs") }];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Sparse-storage predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/issparse.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dense, sparse, integer, logical, container, and resident behavior", location: "crates/runmat-runtime/src/builtins/logical/tests/issparse/tests.rs" }], notes: &[] };
pub(super) const ISSPARSE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("issparse"), slug: Some("issparse"), summary: "Determine whether a value uses sparse storage.", description: "`issparse` returns one logical scalar describing the input's dense or sparse storage representation.", keywords: &["issparse", "sparse", "dense", "storage", "matrix"], related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable) };

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Storage complexity", paragraphs: &["`isreal(A)` reports whether the value uses storage without an imaginary component. It returns one logical scalar rather than an elementwise mask.", "Real numeric, integer, logical, character, and supported duration values return true. Complex storage returns false even when every stored imaginary component is zero. Strings, cells, structs, tables, datetime values, callables, and other objects return false."] },
    BuiltinDocumentationSection { heading: "Sparse and resident values", paragraphs: &["Real and logical sparse matrices return true; complex sparse matrices return false. Resident values are classified from coherent exact-owner storage metadata without downloading their payload."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "real-matrix", title: "Check a real matrix", program: "A = [7 3 2; 2 1 12];\ntf = isreal(A)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "complex-matrix", title: "Detect complex storage", program: "B = [1 3+4i 2; 2i 1 12];\ntf = isreal(B)", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "zero-imaginary", title: "Distinguish complex storage from component values", program: "C = complex(12);\ntf = isreal(C)", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "logical-and-chars", title: "Check logical and character storage", program: "tf_mask = isreal(logical([1 0 1]));\ntf_chars = isreal(['R' 'u' 'n'])", display_output: Some("Both results are true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf_mask);\nassert(tf_chars);" } },
    BuiltinExample { id: "containers", title: "Check text and containers", program: "tf_text = isreal(\"RunMat\");\ntf_cell = isreal({1, 2});\ntf_struct = isreal(struct(\"value\", 1))", display_output: Some("All results are false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf_text);\nassert(~tf_cell);\nassert(~tf_struct);" } },
    BuiltinExample { id: "gpu-storage", title: "Inspect resident storage complexity", program: "G = gpuArray(single([1 2 3]));\nZ = complex(G, zeros(size(G), 'like', G));\ntf_real = isreal(G);\ntf_complex = isreal(Z)", display_output: Some("tf_real = true and tf_complex = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(tf_real);\nassert(~tf_complex);" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `isreal` inspect each element?", answer: "No. It reports the value's storage complexity. Use an imaginary-part comparison for an elementwise test." },
    BuiltinDocumentationFaq { question: "Why is `complex(5)` not real?", answer: "It uses complex storage even though its imaginary component is zero." },
    BuiltinDocumentationFaq { question: "What about logical, duration, or character data?", answer: "Those supported classes do not carry an imaginary component and return true." },
    BuiltinDocumentationFaq { question: "Why do strings and containers return false?", answer: "Their classes are outside the numeric storage-complexity predicate." },
    BuiltinDocumentationFaq { question: "Does a resident query launch a kernel or gather?", answer: "No. RunMat validates and inspects storage metadata." },
];
const RELATED: &[&str] = &[
    "real",
    "imag",
    "complex",
    "isnumeric",
    "islogical",
    "issparse",
    "gpuArray",
    "gather",
];
const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink { label: "real", target: BuiltinDocumentationLinkTarget::Builtin("real") }, BuiltinDocumentationLink { label: "imag", target: BuiltinDocumentationLinkTarget::Builtin("imag") }, BuiltinDocumentationLink { label: "complex", target: BuiltinDocumentationLinkTarget::Builtin("complex") }, BuiltinDocumentationLink { label: "isnumeric", target: BuiltinDocumentationLinkTarget::Builtin("isnumeric") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isreal.rs") }];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Real-storage predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isreal.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Real, complex, sparse, integer, object, and distributed behavior", location: "crates/runmat-runtime/src/builtins/logical/tests/isreal/tests.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU real-storage metadata", location: "crates/runmat-runtime/src/builtins/logical/tests/isreal/tests.rs::wgpu_metadata_matches_storage_complexity" }], notes: &[] };
pub(super) const ISREAL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("isreal"), slug: Some("isreal"), summary: "Determine whether a value uses storage without an imaginary component.", description: "`isreal` returns one logical scalar based on the input's class and real or complex storage kind.", keywords: &["isreal", "real", "complex", "storage", "sparse", "gpuArray"], related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable) };

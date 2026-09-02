use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Comparison semantics", paragraphs: &["`ne(A,B)` and `A ~= B` compare corresponding elements after implicit expansion. A complex pair is unequal when either component differs, and every comparison involving NaN is unequal.", "Numeric, logical, character, string, categorical, and supported object operands use the same domain rules as equality. Fixed-width integers remain exact, characters compare as code points against numeric values, and handle-like values compare by identity."] },
    BuiltinDocumentationSection { heading: "Symbolic and accelerated execution", paragraphs: &["A symbolic operand produces a symbolic inequality with the broadcast result shape. Nonsymbolic comparisons return logical values.", "Compatible resident operands use the provider inequality operation when available. Exact fallback restores explicitly requested residency, and compatible expressions may fuse."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compare different scalars",
        program: "tf = ne(42, 7)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "array",
        title: "Find unequal elements",
        program: "tf = ne([1 2 3 4], [1 0 3 5])",
        display_output: Some("tf = [false true false true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1 0 1])));",
        },
    },
    BuiltinExample {
        id: "implicit-expansion",
        title: "Use implicit expansion",
        program: "tf = ne([1; 2], [1 2 3])",
        display_output: Some("tf = [false true true; true false true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1 1; 1 0 1])));",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Compare complex components",
        program: "tf = ne([1+2i 1+3i], 1+2i)",
        display_output: Some("tf = [false true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1])));",
        },
    },
    BuiltinExample {
        id: "nan",
        title: "Compare NaN",
        program: "tf = ne(NaN, NaN)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "gpu",
        title: "Keep an inequality result resident",
        program:
            "A = gpuArray([1 2 3]);\nB = gpuArray([0 2 4]);\ngtf = ne(A, B);\ntf = gather(gtf)",
        display_output: Some("gtf remains a gpuArray and tf = [true false true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(gtf, 'gpuArray'));\nassert(isequal(tf, logical([1 0 1])));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What class does `ne` return?", answer: "Nonsymbolic comparisons return logical values. Symbolic operands construct symbolic inequalities." }, BuiltinDocumentationFaq { question: "How does `ne` compare NaN?", answer: "`NaN ~= NaN` is true." }, BuiltinDocumentationFaq { question: "How are complex values compared?", answer: "A difference in either component makes the result true." }, BuiltinDocumentationFaq { question: "How are handles compared?", answer: "Handle-like values compare by identity." }, BuiltinDocumentationFaq { question: "Does implicit expansion apply to strings?", answer: "Yes, for compatible string-array shapes." }, BuiltinDocumentationFaq { question: "Are wide integers rounded?", answer: "No. Mixed integer comparisons retain exact fixed-width values." }, BuiltinDocumentationFaq { question: "Can `ne` fuse?", answer: "Compatible elementwise regions may fuse on the selected provider." },
];
const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink { label: "eq", target: BuiltinDocumentationLinkTarget::Builtin("eq") }, BuiltinDocumentationLink { label: "isequal", target: BuiltinDocumentationLinkTarget::Builtin("isequal") }, BuiltinDocumentationLink { label: "lt", target: BuiltinDocumentationLinkTarget::Builtin("lt") }, BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") }, BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/ne/mod.rs") }];
const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink { label: "Inequality runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/ne/mod.rs") }];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: IMPLEMENTATION,
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Numeric, complex, text, identity, broadcast, and error behavior",
            location: "crates/runmat-runtime/src/builtins/logical/rel/ne/tests.rs",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::WgpuTest,
            label: "Actual WGPU comparison and residency",
            location:
                "crates/runmat-runtime/src/builtins/logical/rel/ne/tests.rs::wgpu_matches_host",
        },
    ],
    notes: &[],
};
pub(super) const NE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("ne"), slug: Some("ne"), summary: "Compare values for element-wise inequality.", description: "`ne(A,B)` and `A ~= B` compare compatible operands element by element or construct symbolic inequalities.", keywords: &["ne", "~=", "inequality", "logical", "symbolic inequality", "integer", "complex", "gpuArray"], related: &["eq", "isequal", "lt", "le", "gt", "ge", "gpuArray", "gather"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: Some("Before R2006a"), status: Some(BuiltinDocumentationStatus::Stable) };

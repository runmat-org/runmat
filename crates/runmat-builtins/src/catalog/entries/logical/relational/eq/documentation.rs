use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Comparison semantics", paragraphs: &["`eq(A,B)` and `A == B` compare corresponding elements after implicit expansion. Numeric, logical, character, string, categorical, complex, and supported object operands follow their domain comparison rules. Complex values are equal only when both components match; NaN is unequal to itself.", "All eight fixed-width integer classes compare without first converting authoritative values to `double`. Character data compares by code point against numeric data and as text against string data. Handle-like values compare by runtime identity rather than structural content."] },
    BuiltinDocumentationSection { heading: "Symbolic and accelerated execution", paragraphs: &["A symbolic operand produces a symbolic equation with the broadcast result shape. Ordinary nonsymbolic operands produce logical values.", "Compatible resident operands use the exact provider's equality operation when available. Validated results remain resident; unsupported routes use the exact host comparison and restore explicitly requested residency. Compatible elementwise expressions may fuse."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compare scalars",
        program: "tf = eq(42, 42)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "array",
        title: "Compare arrays element by element",
        program: "A = [1 2 3 4];\nB = [1 0 3 5];\ntf = eq(A, B)",
        display_output: Some("tf = [true false true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 1 0])));",
        },
    },
    BuiltinExample {
        id: "implicit-expansion",
        title: "Use implicit expansion",
        program: "A = [1; 2];\nB = [1 2 3];\ntf = eq(A, B)",
        display_output: Some("tf = [true false false; false true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 0; 0 1 0])));",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Compare both complex components",
        program: "tf = eq([1+2i 1+3i], 1+2i)",
        display_output: Some("tf = [true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0])));",
        },
    },
    BuiltinExample {
        id: "characters",
        title: "Compare character code points",
        program: "tf = eq(['A' 'B' 'C'], 65)",
        display_output: Some("tf = [true false false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 0])));",
        },
    },
    BuiltinExample {
        id: "strings",
        title: "Compare string values",
        program: "names = [\"alice\" \"bob\" \"alice\"];\ntf = eq(names, \"alice\")",
        display_output: Some("tf = [true false true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 1])));",
        },
    },
    BuiltinExample {
        id: "symbolic-equation",
        title: "Construct a symbolic equation",
        program: "syms Y(X)\ncondition = eq(Y(0), 0)",
        display_output: Some("condition = Y(0) == 0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample {
        id: "gpu",
        title: "Keep an equality result resident",
        program:
            "A = gpuArray([1 2 3]);\nB = gpuArray([1 0 3]);\ngtf = eq(A, B);\ntf = gather(gtf)",
        display_output: Some("gtf remains a gpuArray and tf = [true false true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(gtf, 'gpuArray'));\nassert(isequal(tf, logical([1 0 1])));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What class does `eq` return?", answer: "Nonsymbolic comparisons return logical values. Symbolic operands construct symbolic equations." },
    BuiltinDocumentationFaq { question: "How does `eq` compare NaN?", answer: "`NaN == NaN` is false." },
    BuiltinDocumentationFaq { question: "How are complex values compared?", answer: "Both real and imaginary components must match." },
    BuiltinDocumentationFaq { question: "How are handles compared?", answer: "Handle-like values compare by identity." },
    BuiltinDocumentationFaq { question: "Does implicit expansion apply to strings?", answer: "Yes. Compatible string-array shapes expand in the same way as other elementwise comparisons." },
    BuiltinDocumentationFaq { question: "Are wide integers rounded?", answer: "No. Mixed integer comparisons retain exact fixed-width values." },
    BuiltinDocumentationFaq { question: "Does a GPU comparison gather?", answer: "A supported same-owner operation stays resident. Exact fallback restores explicitly requested residency." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "ne", target: BuiltinDocumentationLinkTarget::Builtin("ne") },
    BuiltinDocumentationLink { label: "isequal", target: BuiltinDocumentationLinkTarget::Builtin("isequal") },
    BuiltinDocumentationLink { label: "lt", target: BuiltinDocumentationLinkTarget::Builtin("lt") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/eq/mod.rs") },
];
const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink { label: "Equality runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/eq/mod.rs") }];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: IMPLEMENTATION,
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Numeric, complex, text, symbolic, identity, broadcast, and error behavior",
            location: "crates/runmat-runtime/src/builtins/logical/rel/eq/tests.rs",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::WgpuTest,
            label: "Actual WGPU comparison and residency",
            location:
                "crates/runmat-runtime/src/builtins/logical/rel/eq/tests.rs::wgpu_matches_host",
        },
    ],
    notes: &[],
};
pub(super) const EQ_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("eq"), slug: Some("eq"), summary: "Compare values for element-wise equality.", description: "`eq(A,B)` and `A == B` compare compatible operands element by element or construct symbolic equations.", keywords: &["eq", "==", "equality", "logical", "symbolic equation", "integer", "complex", "gpuArray"], related: &["ne", "isequal", "lt", "le", "gt", "ge", "gpuArray", "gather"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: Some("Before R2006a"), status: Some(BuiltinDocumentationStatus::Stable) };

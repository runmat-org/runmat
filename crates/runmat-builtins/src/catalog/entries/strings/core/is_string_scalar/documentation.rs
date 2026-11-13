use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`isStringScalar(value)` returns scalar logical true for a string scalar, including a string array containing exactly one element. It returns false for character arrays, non-scalar string arrays, numeric values, cells, structs, and other values.",
        "The predicate inspects the value's kind and string-array element count. Numeric and provider-resident inputs return false without payload reads or provider access.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &["This metadata-only predicate runs in native and browser runtimes without a GPU kernel, fusion, transfer, callback, workspace, filesystem, network, randomness, or time effect. At most one output is supported."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-example",
        title: "Test a scalar string",
        program: "tf = isStringScalar(\"runmat\")",
        display_output: Some("tf = logical 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "nonscalar",
        title: "Reject a non-scalar string array",
        program: "tf = isStringScalar([\"run\" \"mat\"])",
        display_output: Some("tf = logical 0"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, false));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("isStringScalar"), slug: Some("isStringScalar"),
    summary: "Return true for a scalar MATLAB string.", description: "`isStringScalar(value)` returns a scalar logical type-test result.",
    keywords: &["isStringScalar", "strings", "core", "matlab", "compatibility"], related: &["string", "char"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/is_string_scalar.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/is_string_scalar.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "scalar string and nontext inputs", location: "builtins::strings::core::compat::tests" }], notes: &["Legacy title, summary, signature, example, two related links and source reconciled. The legacy GPU-gather note was wrong: resident numeric values return false without transfer."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

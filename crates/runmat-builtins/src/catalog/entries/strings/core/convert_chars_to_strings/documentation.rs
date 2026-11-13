use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Conversion", paragraphs: &[
        "`convertCharsToStrings` converts a character array to one string scalar. In RunMat's current implementation, characters are read column by column from the array. A cell array converts to a same-shape string array only when every element is a character vector; otherwise the entire cell value passes through unchanged.",
        "Numeric values and resident handles pass through with class, shape, and provider ownership unchanged. No nested cell or struct traversal occurs.",
    ] },
    BuiltinDocumentationSection { heading: "Outputs and placement", paragraphs: &[
        "Multiple inputs convert independently. If more outputs are requested than inputs, RunMat fills the extra outputs with scalar double zero. Unrequested outputs are discarded, but every input is still converted.",
        "Conversion materializes host strings; non-target resident values do not gather. There is no GPU kernel, fusion, callback, workspace, filesystem, network, randomness, or time effect. The builtin runs in native and browser runtimes.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-character-matrix",
        title: "Convert a character matrix",
        program: "s = convertCharsToStrings(['ab'; 'cd'])",
        display_output: Some("s = \"acbd\""),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(s, \"acbd\"));",
        },
    },
    BuiltinExample {
        id: "cellstr",
        title: "Convert a cell array of character rows",
        program: "S = convertCharsToStrings({'one', 'two'})",
        display_output: Some("S is a 1-by-2 string array"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(S), [1 2])); assert(isequal(S(2), \"two\"));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("convertCharsToStrings"), slug: Some("convertCharsToStrings"),
    summary: "Convert character arrays and cellstr values to string arrays.", description: "Convert character arrays to strings and cellstr containers to same-shape string arrays.",
    keywords: &["convertCharsToStrings", "strings", "core", "matlab", "compatibility"], related: &["string", "char"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/convert_chars_to_strings.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/convert_chars_to_strings.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "character and cellstr conversion", location: "builtins::strings::core::compat::tests" }], notes: &["Sidecar signature, behavior, example, links, and source reconciled. Column-order and extra-output padding are documented as existing RunMat behavior without claiming MATLAB parity."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

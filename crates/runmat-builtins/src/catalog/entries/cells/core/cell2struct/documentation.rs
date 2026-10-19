use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "row-to-scalar", title: "Create a scalar structure from a row", program: "S = cell2struct({1, 'entry'}, {'id', 'name'}, 2);", display_output: Some("S has fields id and name"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(S.id == 1);\nassert(strcmp(S.name, 'entry'));" } },
    BuiltinExample { id: "default-dimension", title: "Use the first dimension by default", program: "S = cell2struct({10; 20}, {'low'; 'high'});", display_output: Some("S.low = 10 and S.high = 20"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(S.low == 10);\nassert(S.high == 20);" } },
    BuiltinExample { id: "string-fields", title: "Supply field names as strings", program: "S = cell2struct({3, 4}, [\"left\", \"right\"], 2);", display_output: Some("S has fields left and right"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(S.left == 3);\nassert(S.right == 4);" } },
    BuiltinExample { id: "character-matrix-fields", title: "Read field names from character rows", program: "names = char('red', 'blu');\nS = cell2struct({1; 2}, names, 1);", display_output: Some("S.red = 1 and S.blu = 2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(S.red == 1);\nassert(S.blu == 2);" } },
    BuiltinExample { id: "structure-array", title: "Create a structure array", program: "S = cell2struct({1, 2; 10, 20}, {'x'; 'y'}, 1);", display_output: Some("S is a 1-by-2 structure array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(S), [1 2]));\nassert(getfield(S, {1}, 'x') == 1);\nassert(getfield(S, {2}, 'y') == 20);" } },
    BuiltinExample { id: "exact-integers", title: "Preserve fixed-width integer payloads", program: "wide = uint64(2^53) + uint64(1);\nS = cell2struct({wide; int8(-7)}, {'wide'; 'small'});", display_output: Some("Fields retain uint64 and int8 storage"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(S.wide, 'uint64'));\nassert(S.wide == wide);\nassert(isa(S.small, 'int8'));" } },
];

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Conversion", paragraphs: &["`cell2struct(C, fields, dim)` assigns entries along dimension `dim` of `C` to the supplied field names. All remaining dimensions become the structure-array shape. The dimension defaults to 1.", "Field names may be a character vector, the rows of a character matrix, a string scalar or array, or a cell array of text scalars. The field count must equal the selected dimension extent."] },
    BuiltinDocumentationSection { heading: "Values and structure arrays", paragraphs: &["A one-element output is returned as a scalar structure. Nonscalar and empty results retain the remaining dimensions and ordered field schema as structure arrays.", "Cell payloads move into their fields without numeric conversion. Fixed-width integers remain exact, and nested resident handles retain their provider ownership without a gather or kernel."] },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What happens to the selected dimension?", answer: "It becomes a singleton dimension; the other dimensions define the structure-array shape." },
    BuiltinDocumentationFaq { question: "Does the result remain a cell array?", answer: "No. The selected cell values become fields of a scalar structure or typed structure array." },
    BuiltinDocumentationFaq { question: "Are integer or GPU-resident field values converted?", answer: "No. Values move into fields unchanged; nested resident handles are not gathered." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "num2cell",
        target: BuiltinDocumentationLinkTarget::Builtin("num2cell"),
    },
    BuiltinDocumentationLink {
        label: "struct",
        target: BuiltinDocumentationLinkTarget::Builtin("struct"),
    },
    BuiltinDocumentationLink {
        label: "cell",
        target: BuiltinDocumentationLinkTarget::Builtin("cell"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/cells/core/cell2struct") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dimensions, storage order, exact payloads, and validation", location: "builtins::cells::core::cell2struct::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed output facts and diagnostics", location: "catalog::entries::cells::core::cell2struct::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cell2struct"), slug: Some("cell2struct"),
    summary: "Convert a cell array into a structure or structure array.",
    description: "`cell2struct` maps one cell dimension to named fields and retains the remaining dimensions as the output shape.",
    keywords: &["cell2struct", "cell", "struct", "structure array", "conversion"],
    related: &["num2cell", "struct", "cell"], sections: SECTIONS, examples: EXAMPLES,
    example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE,
    introduced: Some("before R2006a"), status: Some(BuiltinDocumentationStatus::Stable),
};

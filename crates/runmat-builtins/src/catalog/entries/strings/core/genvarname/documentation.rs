use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"Character row-vector input returns a character row vector.",
"String scalar input returns a string scalar.",
"String arrays preserve their shape and return string arrays.",
"Cell arrays of character vectors or string scalars preserve their shape and return cell arrays of character row vectors.",
"Character array input must be a character vector; use a string array or cell array to generate multiple names.",
"Whitespace is removed and the following alphabetic character is capitalized.",
"Nonalphanumeric, non-underscore characters are replaced by their hexadecimal code.",
"Names that do not start with a letter are prefixed with `x`, and MATLAB keywords are repaired with the legacy `xKeyword` form such as `xFor`.",
"Names are truncated to the current `namelengthmax` compatibility length.",
"Duplicate generated names and names present in `exclusions` receive numeric suffixes.",
"Suffixes are assigned in MATLAB column-major linear order, including N-D cell and string containers. Character input must have at most one row and no non-singleton dimensions beyond two; character output length may change.",
"Names and exclusions must contain host text. Numeric, integer and resident values reject directly, including invalid nested containers; no provider lookup or download occurs. This corrects the former sidecar claim that GPU inputs were gathered.",
] },
BuiltinDocumentationSection { heading: "Syntax", paragraphs: &[
"varname = genvarname(str)\nvarname = genvarname(str, exclusions)",
"`str` can be a character array, string scalar/array, or cell array of character vectors/string scalars.",
"`exclusions` can be text or a text container of names that generated results must avoid.",
"The output preserves the input text container shape where applicable.",
] },
BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
"This portable host operation is pure apart from errors and never suspends. It performs no callbacks, workspace changes, filesystem, randomness, time services, provider hooks, or GPU fusion.",
"Interpreter and compiled calls share the default runtime binding. Zero requested outputs evaluates and discards the return value; more than one output is rejected before processing. Distributed input is unsupported. Device kernels and provider benchmarks do not apply to host-only text and symbolic values.",
] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Generate Unique Names",
        program: "v = genvarname({'A','A','A','A'})",
        display_output: Some("v = {'A','A1','A2','A3'}"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(v, {'A','A1','A2','A3'}));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Avoid Excluded Names",
        program: "v = genvarname('A', {'A','A1'})",
        display_output: Some("v = 'A2'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(v)); assert(strcmp(v,'A2'));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Normalize Invalid Text",
        program: "v = genvarname('1 a-b')",
        display_output: Some("v = 'x1A0x2Db'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(v)); assert(strcmp(v,'x1A0x2Db'));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog, title: Some("genvarname"), slug: Some("genvarname"),
 summary: "Construct valid, unique MATLAB variable names from text.", description: "`genvarname(str)` converts text into valid MATLAB variable names and makes names unique within the returned set. `genvarname(str, exclusions)` also avoids names listed in `exclusions`.",
 keywords: &["genvarname","variable name","identifier","legacy","makeValidName","makeUniqueStrings"], related: &["isvarname","cellstr","string"],
 sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
 faqs: &[
 BuiltinDocumentationFaq { question: "Is `genvarname` a modern MATLAB API?", answer: "No. MATLAB marks `genvarname` for removal and recommends `matlab.lang.makeValidName` with `matlab.lang.makeUniqueStrings`. RunMat supports it for backward compatibility with older MATLAB code." },
BuiltinDocumentationFaq { question: "Does `genvarname` create variables?", answer: "No. It only returns text that can be used as variable names." },
BuiltinDocumentationFaq { question: "Are exclusions normalized first?", answer: "No. Exclusions are treated as already-specified names to avoid, matching the documented role of excluding existing variable names." },
 ],
 links: &[BuiltinDocumentationLink { label: "isvarname", target: BuiltinDocumentationLinkTarget::Builtin("isvarname") },
BuiltinDocumentationLink { label: "cellstr", target: BuiltinDocumentationLinkTarget::Builtin("cellstr") },
BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/genvarname.rs") }], media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/genvarname.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "unit", location: "builtins::strings::core::genvarname::tests" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Identity inference tests", location: "crates/runmat-builtins/src/catalog/entries/strings/core/genvarname/inference.rs" }],
 notes: &["Legacy behaviors, syntax, examples, FAQs and links are reconciled here. Typed placement replaces GPU/fusion metadata. Example assertions verify semantic values rather than presentation. Integrated native, browser and documentation execution remains required; catalog metadata alone is not runtime evidence."],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

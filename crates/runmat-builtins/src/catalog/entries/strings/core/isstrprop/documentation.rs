use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`isstrprop(text, category)` classifies characters by `alpha`, `alphanum`, `digit`, `xdigit`, `wspace`, `upper`, `lower`, `punct`, `cntrl`, `graphic`, or `print`. Category may be a character vector or string scalar.",
        "Character input returns a same-shaped logical array. String arrays and cell containers produce cell arrays of logical character classifications. Numeric values, including all eight integer classes, are interpreted as Unicode character codes; integer values are read exactly and clamped to the BMP code-unit range before classification.",
        "The optional `\"ForceCellOutput\", flag` pair wraps a non-cell result in a scalar cell when flag is true. Flag accepts a one-element logical value or a real double scalar or dense array equal to zero or one. Single, integer, complex, sparse, and GPU-resident flags are not supported. Invalid categories, options, flags, and unsupported text values error.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and compatibility", paragraphs: &[
        "Classification runs on the host in native and browser runtimes. Automatic resident numeric input is downloaded exactly; explicit resident input is a RunMat-only form and is rejected in strict MATLAB compatibility mode before provider access. A supported download or property gather can suspend.",
        "No GPU classification kernel or fusion form is registered. The binding can throw but has no callback, workspace, filesystem, network, randomness, or time effect. One output is supported.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "legacy-example",
        title: "Classify letters",
        program: "tf = isstrprop(\"A1\", \"alpha\")",
        display_output: Some("tf = [true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0])));",
        },
    },
    BuiltinExample {
        id: "integer-codes",
        title: "Classify exact integer character codes",
        program: "tf = isstrprop(uint16([65 49]), \"alpha\")",
        display_output: Some("tf = [true false]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0])));",
        },
    },
    BuiltinExample {
        id: "force-cell",
        title: "Wrap a result in a cell",
        program: "C = isstrprop(\"A1\", \"alpha\", \"ForceCellOutput\", true)",
        display_output: Some("C is a scalar cell of logical values"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(iscell(C)); assert(isequal(C{1}, logical([1 0])));",
        },
    },
    BuiltinExample {
        id: "force-cell-double-flag",
        title: "Use the exact Double zero/one flag with integer character codes",
        program: "C = isstrprop(uint16([65 49]), \"alpha\", \"ForceCellOutput\", 1)",
        display_output: Some("C is a scalar cell containing [true false]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(iscell(C)); assert(isequal(size(C), [1 1])); assert(isequal(C{1}, logical([1 0])));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("isstrprop"), slug: Some("isstrprop"),
    summary: "Classify characters in text by character property.", description: "`isstrprop(text, category)` returns logical character-property classifications, optionally wrapped in a cell.",
    keywords: &["isstrprop", "strings", "core", "matlab", "compatibility"], related: &["string", "char", "isletter", "isspace"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What categories are supported?", answer: "alpha, alphanum, digit, xdigit, wspace, upper, lower, punct, cntrl, graphic, and print." },
        BuiltinDocumentationFaq { question: "Are integer codes converted through double?", answer: "No. Fixed-width integer values are decoded from native storage before the character-property test." },
        BuiltinDocumentationFaq { question: "What does ForceCellOutput do?", answer: "When true, it wraps a non-cell logical result in a scalar cell; an existing cell result remains a cell." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/isstrprop.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/isstrprop.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "integer and extension behavior", location: "builtins::strings::core::compat::tests" }], notes: &["Legacy sidecar title, summary, signature, example, related links and source link retained with source updated to its extracted runtime module; old signature was incomplete and is split into explicit base and ForceCellOutput forms. Its blanket GPU-gather note is narrowed to the actual numeric resident route."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

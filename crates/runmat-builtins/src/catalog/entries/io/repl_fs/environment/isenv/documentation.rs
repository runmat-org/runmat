use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "defined", title: "Test a defined variable", program: "setenv('RUNMAT_DOC_ISENV_DEFINED', 'yes');\ntf = isenv('RUNMAT_DOC_ISENV_DEFINED');\nunsetenv('RUNMAT_DOC_ISENV_DEFINED');", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "undefined", title: "Test an undefined variable", program: "unsetenv('RUNMAT_DOC_ISENV_UNDEFINED');\ntf = isenv('RUNMAT_DOC_ISENV_UNDEFINED');", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "array", title: "Test several names", program: "setenv('RUNMAT_DOC_ISENV_A', '');\nunsetenv('RUNMAT_DOC_ISENV_B');\ntf = isenv([\"RUNMAT_DOC_ISENV_A\", \"RUNMAT_DOC_ISENV_B\"]);\nunsetenv('RUNMAT_DOC_ISENV_A');", display_output: Some("tf = [true false] on Unix-like and browser sessions"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Browser, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, [true false]));" } },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "getenv",
        target: BuiltinDocumentationLinkTarget::Builtin("getenv"),
    },
    BuiltinDocumentationLink {
        label: "setenv",
        target: BuiltinDocumentationLinkTarget::Builtin("setenv"),
    },
    BuiltinDocumentationLink {
        label: "unsetenv",
        target: BuiltinDocumentationLinkTarget::Builtin("unsetenv"),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isenv"),
    slug: Some("isenv"),
    summary: "Test whether environment variables are defined.",
    description: "`isenv(name)` returns logical results with the name container's shape without reading the variables' values.",
    keywords: &["isenv", "environment variable", "defined"],
    related: &["getenv", "setenv", "unsetenv"],
    sections: &[BuiltinDocumentationSection {
        heading: "Definition and shape",
        paragraphs: &[
            "A character row or string scalar produces one logical scalar. String arrays and cell arrays of character rows produce a logical array with the same dimensions.",
            "An empty environment value is still defined on Unix-like systems and in browser sessions. Windows removes an environment variable when it is assigned an empty value, so the result follows the host platform.",
        ],
    }],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does isenv read the variable's value?",
        answer: "No. It tests whether the name exists. Use `getenv` to read the value.",
    }],
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Environment predicate runtime",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/environment/isenv/mod.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Scalar and shaped-container existence checks",
                location: "builtins::io::repl_fs::environment::isenv::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Executable native and browser examples",
                location: "scripts/runtime/verify-builtin-examples.mjs",
            },
        ],
        notes: &[],
    },
    introduced: Some("R2022b"),
    status: Some(BuiltinDocumentationStatus::Stable),
};

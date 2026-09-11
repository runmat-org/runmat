use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "remove", title: "Remove a variable", program: "setenv('RUNMAT_DOC_UNSETENV', 'temporary');\nunsetenv('RUNMAT_DOC_UNSETENV');\ntf = isenv('RUNMAT_DOC_UNSETENV');", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "missing", title: "Remove an undefined name", program: "unsetenv('RUNMAT_DOC_UNSETENV_MISSING');\nunsetenv('RUNMAT_DOC_UNSETENV_MISSING');\ntf = isenv('RUNMAT_DOC_UNSETENV_MISSING');", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "several", title: "Remove several variables", program: "setenv([\"RUNMAT_DOC_UNSETENV_A\", \"RUNMAT_DOC_UNSETENV_B\"], \"value\");\nunsetenv([\"RUNMAT_DOC_UNSETENV_A\", \"RUNMAT_DOC_UNSETENV_B\"]);\ntf = isenv([\"RUNMAT_DOC_UNSETENV_A\", \"RUNMAT_DOC_UNSETENV_B\"]);", display_output: Some("tf = [false false]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, [false false]));" } },
    BuiltinExample { id: "status-extension", title: "Request a RunMat status", program: "setenv('RUNMAT_DOC_UNSETENV_STATUS', 'temporary');\nstatus = unsetenv('RUNMAT_DOC_UNSETENV_STATUS');", display_output: Some("status = 0"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0);" } },
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
        label: "isenv",
        target: BuiltinDocumentationLinkTarget::Builtin("isenv"),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("unsetenv"),
    slug: Some("unsetenv"),
    summary: "Remove environment variables from the current RunMat session.",
    description: "`unsetenv(name)` removes one or more names from the environment visible to RunMat and subsequently launched child processes.",
    keywords: &["unsetenv", "environment variable", "remove"],
    related: &["getenv", "setenv", "isenv"],
    sections: &[BuiltinDocumentationSection {
        heading: "Removal and shape",
        paragraphs: &[
            "A character row or string scalar names one variable. String arrays and cell arrays of character rows remove each corresponding name. A name that is already absent has no effect.",
            "The documented form has no output. RunMat mode can additionally return status `0` after a valid removal or status `1` for an invalid name. Inputs are validated as one batch before any variable is removed.",
        ],
    }],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Is it an error when the name is already absent?",
            answer: "No. Removing an undefined variable has no effect.",
        },
        BuiltinDocumentationFaq {
            question: "Does unsetenv affect the parent shell?",
            answer: "No. It changes only this RunMat process and child processes launched afterward.",
        },
    ],
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Environment removal runtime",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/environment/unsetenv/mod.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Scalar, container, idempotence, validation, and output behavior",
                location: "builtins::io::repl_fs::environment::unsetenv::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Executable native and browser examples",
                location: "scripts/runtime/verify-builtin-examples.mjs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

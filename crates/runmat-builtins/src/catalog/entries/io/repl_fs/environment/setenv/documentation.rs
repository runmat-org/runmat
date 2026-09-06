use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "set", title: "Set a variable", program: "setenv('RUNMAT_DOC_SETENV', 'development');\nvalue = getenv('RUNMAT_DOC_SETENV');\nunsetenv('RUNMAT_DOC_SETENV');", display_output: Some("value = 'development'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(value, 'development'));" } },
    BuiltinExample { id: "replace", title: "Replace an existing value", program: "setenv('RUNMAT_DOC_SETENV_REPLACE', 'first');\nsetenv('RUNMAT_DOC_SETENV_REPLACE', 'second');\nvalue = getenv('RUNMAT_DOC_SETENV_REPLACE');\nunsetenv('RUNMAT_DOC_SETENV_REPLACE');", display_output: Some("value = 'second'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(value, 'second'));" } },
    BuiltinExample { id: "several", title: "Set several variables", program: "setenv([\"RUNMAT_DOC_SETENV_A\", \"RUNMAT_DOC_SETENV_B\"], [\"alpha\", \"beta\"]);\nvalues = getenv([\"RUNMAT_DOC_SETENV_A\", \"RUNMAT_DOC_SETENV_B\"]);\nunsetenv([\"RUNMAT_DOC_SETENV_A\", \"RUNMAT_DOC_SETENV_B\"]);", display_output: Some("values = [\"alpha\" \"beta\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(values, [\"alpha\", \"beta\"]));" } },
    BuiltinExample { id: "numeric", title: "Set a numeric scalar value", program: "setenv('RUNMAT_DOC_SETENV_INTEGER', intmax('uint64'));\nvalue = getenv('RUNMAT_DOC_SETENV_INTEGER');\nunsetenv('RUNMAT_DOC_SETENV_INTEGER');", display_output: Some("value = '18446744073709551615'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(value, '18446744073709551615'));" } },
    BuiltinExample { id: "dictionary", title: "Set values from a dictionary", program: "environment = dictionary([\"RUNMAT_DOC_SETENV_C\", \"RUNMAT_DOC_SETENV_D\"], [\"one\", \"two\"]);\nsetenv(environment);\nvalues = getenv([\"RUNMAT_DOC_SETENV_C\", \"RUNMAT_DOC_SETENV_D\"]);\nunsetenv([\"RUNMAT_DOC_SETENV_C\", \"RUNMAT_DOC_SETENV_D\"]);", display_output: Some("values = [\"one\" \"two\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(values, [\"one\", \"two\"]));" } },
    BuiltinExample { id: "status-extension", title: "Request RunMat status outputs", program: "[status, message] = setenv('RUNMAT_DOC_SETENV_STATUS', 'ready');\nvalue = getenv('RUNMAT_DOC_SETENV_STATUS');\nunsetenv('RUNMAT_DOC_SETENV_STATUS');", display_output: Some("status = 0 and message is empty"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0); assert(isempty(message)); assert(strcmp(value, 'ready'));" } },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "getenv",
        target: BuiltinDocumentationLinkTarget::Builtin("getenv"),
    },
    BuiltinDocumentationLink {
        label: "isenv",
        target: BuiltinDocumentationLinkTarget::Builtin("isenv"),
    },
    BuiltinDocumentationLink {
        label: "unsetenv",
        target: BuiltinDocumentationLinkTarget::Builtin("unsetenv"),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("setenv"),
    slug: Some("setenv"),
    summary: "Set environment variables for the current RunMat session.",
    description: "`setenv` validates a complete update batch, then changes the environment visible to RunMat and subsequently launched child processes.",
    keywords: &["setenv", "environment variable", "process environment"],
    related: &["getenv", "isenv", "unsetenv", "system"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Names, values, and batches",
            paragraphs: &[
                "Names may be character rows, strings, string arrays, or cell arrays of character rows. Values may be text, scalar numeric values, matching containers, or scalar values broadcast across a name container. A dictionary supplies names as keys and values as dictionary values.",
                "Calling `setenv(name)` assigns an empty value. A missing string value removes the name. Assignment of an empty value follows the platform: Unix-like systems and browser sessions retain a defined empty variable, while Windows removes it.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Scope and execution",
            paragraphs: &[
                "Changes affect the current RunMat session and child processes launched afterward; they cannot change the parent shell. A batch is fully decoded and validated before any mutation, so an invalid later element cannot leave an earlier update applied.",
                "The documented forms do not return a value. RunMat mode can additionally return a numeric status and diagnostic message. Environment mutation is host-owned and never gathers numeric values merely to reinterpret them as names.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How do I remove a variable?",
            answer: "Use `unsetenv(name)` or provide a missing string value to `setenv`. An ordinary empty value remains defined on platforms that support empty environment values.",
        },
        BuiltinDocumentationFaq {
            question: "Does the change persist after RunMat exits?",
            answer: "No. The process environment is not a persistent system configuration store.",
        },
        BuiltinDocumentationFaq {
            question: "Are updates atomic?",
            answer: "Input decoding and validation are atomic. Host environment APIs do not provide a transactional multi-variable commit, so an unexpected host failure may occur after earlier valid updates have been applied.",
        },
    ],
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Environment mutation runtime",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/environment/setenv/mod.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Scalar, batch, dictionary, numeric, output, and failure behavior",
                location: "builtins::io::repl_fs::environment::setenv::tests",
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

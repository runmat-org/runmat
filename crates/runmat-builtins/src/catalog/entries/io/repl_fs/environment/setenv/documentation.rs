use crate::*;

mod examples;
use examples::EXAMPLES;

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

    summary: "Set environment variables visible to RunMat.",

    description: "`setenv` validates a complete update batch, then changes the environment visible to RunMat and subsequently launched child processes.",

    keywords: &["setenv",
 "environment variable",
 "process environment"],

    related: &["getenv",
 "isenv",
 "unsetenv",
 "system"],

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
                "Native changes affect the current process and child processes launched afterward, not the parent shell. Browser changes affect the runtime instance's in-memory environment. A batch is fully decoded and validated before any mutation, so an invalid later element cannot leave an earlier update applied.",

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

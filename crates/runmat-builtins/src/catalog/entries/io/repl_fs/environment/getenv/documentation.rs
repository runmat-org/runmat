use crate::*;

mod examples;
use examples::EXAMPLES;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
 heading: "Read environment variables",
 paragraphs: &["`getenv(name)` reads a variable visible to the current RunMat process. A missing scalar variable returns an empty character row. Character and string scalar queries return character rows; string arrays and cell arrays preserve their container type and dimensions.",
 "`getenv()` returns all visible names and values in a string dictionary. Names retain their exact spelling, including punctuation that cannot be used as a structure field name."] },

    BuiltinDocumentationSection {
 heading: "Platforms and execution",
 paragraphs: &["Name matching follows the host platform: Windows environment names are case-insensitive, while Unix-like systems and browser sessions use case-sensitive names. The browser runtime provides an in-memory environment shared by calls in that runtime instance, with the same query and mutation interface.",
 "Environment access is host-owned and does not use an acceleration provider. Numeric and provider-resident inputs reject before any lookup. RunMat mode additionally accepts padded multirow character matrices and string scalars inside cells."] },

];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
 question: "What is returned for an undefined variable?",
 answer: "A scalar query returns an empty character row. Nonscalar string and cell queries retain their input container and place an empty value at the corresponding element." },

    BuiltinDocumentationFaq {
 question: "Does getenv modify the environment?",
 answer: "No. Use `setenv` to set a value and `unsetenv` to remove one." },

    BuiltinDocumentationFaq {
 question: "Can a GPU accelerate the lookup?",
 answer: "No. Environment access belongs to the process or browser session and accepts text names only." },

];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
 label: "setenv",
 target: BuiltinDocumentationLinkTarget::Builtin("setenv") },

    BuiltinDocumentationLink {
 label: "isenv",
 target: BuiltinDocumentationLinkTarget::Builtin("isenv") },

    BuiltinDocumentationLink {
 label: "unsetenv",
 target: BuiltinDocumentationLinkTarget::Builtin("unsetenv") },

    BuiltinDocumentationLink {
 label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/environment/getenv/mod.rs") },

];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("getenv"),

    slug: Some("getenv"),

    summary: "Read environment variables visible to RunMat.",

    description: "`getenv` reads one or more environment variables or returns the complete visible environment as a dictionary.",

    keywords: &["getenv",
 "environment variable",
 "process environment"],

    related: &["setenv",
 "isenv",
 "unsetenv",
 "path",
 "tempdir"],

    sections: SECTIONS,

    examples: EXAMPLES,

    example_exemption: None,

    faqs: FAQS,

    links: LINKS,

    media: &[],

    evidence: BuiltinDocumentationEvidence {

        implementation: &[BuiltinDocumentationLink {

            label: "Environment query runtime",

            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/environment/getenv/mod.rs"),

        }],

        verification: &[
            BuiltinEvidenceReference {

                kind: BuiltinEvidenceKind::UnitTest,

                label: "Scalar, container, dictionary, compatibility, and browser behavior",

                location: "builtins::io::repl_fs::environment::getenv::tests",

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

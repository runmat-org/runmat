use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Read environment variables", paragraphs: &["`getenv(name)` reads a variable visible to the current RunMat process. A missing scalar variable returns an empty character row. Character and string scalar queries return character rows; string arrays and cell arrays preserve their container type and dimensions.", "`getenv()` returns all visible names and values in a string dictionary. Names retain their exact spelling, including punctuation that cannot be used as a structure field name."] },
    BuiltinDocumentationSection { heading: "Platforms and execution", paragraphs: &["Name matching follows the host platform: Windows environment names are case-insensitive, while Unix-like systems and browser sessions use case-sensitive names. The browser runtime provides a session-local environment with the same query and mutation interface.", "Environment access is host-owned and does not use an acceleration provider. Numeric and provider-resident inputs reject before any lookup. RunMat mode additionally accepts padded multirow character matrices and string scalars inside cells."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "scalar", title: "Read one variable", program: "setenv('RUNMAT_DOC_GETENV_SCALAR', 'ready');\nvalue = getenv('RUNMAT_DOC_GETENV_SCALAR');\nunsetenv('RUNMAT_DOC_GETENV_SCALAR');", display_output: Some("value = 'ready'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(value, 'ready'));" } },
    BuiltinExample { id: "missing", title: "Handle an undefined variable", program: "unsetenv('RUNMAT_DOC_GETENV_MISSING');\nvalue = getenv('RUNMAT_DOC_GETENV_MISSING');", display_output: Some("value is an empty character row"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(value));" } },
    BuiltinExample { id: "string-array", title: "Read several names", program: "setenv('RUNMAT_DOC_GETENV_A', 'alpha');\nsetenv('RUNMAT_DOC_GETENV_B', 'beta');\nvalues = getenv([\"RUNMAT_DOC_GETENV_A\", \"RUNMAT_DOC_GETENV_B\"]);\nunsetenv([\"RUNMAT_DOC_GETENV_A\", \"RUNMAT_DOC_GETENV_B\"]);", display_output: Some("values = [\"alpha\" \"beta\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(values, [\"alpha\", \"beta\"]));" } },
    BuiltinExample { id: "cell", title: "Preserve a cell array", program: "setenv('RUNMAT_DOC_GETENV_C', 'first');\nsetenv('RUNMAT_DOC_GETENV_D', 'second');\nvalues = getenv({'RUNMAT_DOC_GETENV_C', 'RUNMAT_DOC_GETENV_D'});\nunsetenv({'RUNMAT_DOC_GETENV_C', 'RUNMAT_DOC_GETENV_D'});", display_output: Some("values is a 1-by-2 cell array of character rows"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(values), [1 2])); assert(strcmp(values{1}, 'first')); assert(strcmp(values{2}, 'second'));" } },
    BuiltinExample { id: "all", title: "Inspect the environment dictionary", program: "setenv('RUNMAT_DOC_GETENV_ALL', 'visible');\nenvironment = getenv();\nvalue = environment(\"RUNMAT_DOC_GETENV_ALL\");\nunsetenv('RUNMAT_DOC_GETENV_ALL');", display_output: Some("value = \"visible\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(value == \"visible\");" } },
    BuiltinExample { id: "runmat-cell-string", title: "Use string values inside a cell in RunMat mode", program: "setenv('RUNMAT_DOC_GETENV_STRING_CELL', 'value');\nvalues = getenv({\"RUNMAT_DOC_GETENV_STRING_CELL\"});\nunsetenv('RUNMAT_DOC_GETENV_STRING_CELL');", display_output: Some("values{1} = \"value\""), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(values{1} == \"value\");" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What is returned for an undefined variable?", answer: "A scalar query returns an empty character row. Nonscalar string and cell queries retain their input container and place an empty value at the corresponding element." },
    BuiltinDocumentationFaq { question: "Does getenv modify the environment?", answer: "No. Use `setenv` to set a value and `unsetenv` to remove one." },
    BuiltinDocumentationFaq { question: "Can a GPU accelerate the lookup?", answer: "No. Environment access belongs to the process or browser session and accepts text names only." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "setenv", target: BuiltinDocumentationLinkTarget::Builtin("setenv") },
    BuiltinDocumentationLink { label: "isenv", target: BuiltinDocumentationLinkTarget::Builtin("isenv") },
    BuiltinDocumentationLink { label: "unsetenv", target: BuiltinDocumentationLinkTarget::Builtin("unsetenv") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/environment/getenv/mod.rs") },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("getenv"),
    slug: Some("getenv"),
    summary: "Read environment variables visible to the current RunMat session.",
    description: "`getenv` reads one or more environment variables or returns the complete visible environment as a dictionary.",
    keywords: &["getenv", "environment variable", "process environment"],
    related: &["setenv", "isenv", "unsetenv", "path", "tempdir"],
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

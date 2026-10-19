use super::examples::EXAMPLES;
use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("exist"),
    slug: Some("exist"),
    summary: "Return the documented numeric code for a variable, file, folder, builtin, model, or class.",
    description: "`exist` searches the active workspace and configured search path and returns a host double scalar code from 0 through 8. The optional documented searchType values are `builtin`, `class`, `dir`, `file`, and `var`.",
    keywords: &[
        "exist",
        "variable",
        "file",
        "folder",
        "builtin",
        "class"
    ],
    related: &[
        "isfile",
        "isfolder",
        "which"
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Queries and return codes",
            paragraphs: &[
                "The documented return codes are 0 not found or inaccessible, 1 variable, 2 general file, 3 MEX file, 4 loaded or on-path Simulink model, 5 builtin, 6 P-code, 7 folder, and 8 class.",
                "Without searchType, variables and builtins have highest precedence; folders then precede classes and files; remaining file kinds follow function precedence within search-path order.",
                "`exist(name,'file')` searches files and folders and can therefore return 2, 3, 4, 6, 7, or 0.",
                "`name` and documented searchType values are host string scalars or character vectors. Integer and logical inputs are inapplicable and reject without conversion.",
                "RunMat supports additional selector aliases and specialized selectors for its own runtime. They are an explicit RunMat-only extension and reject in MATLAB compatibility mode.",
                "`exist(handleObject,'handle')` with a nontext first argument is a separate RunMat-only extension.",
                "Arbitrary unauthenticated HTTP and HTTPS URL lookup, JVM class discovery, and loaded Simulink runtime state remain environment-level gaps. Local and configured RunMat search-path lookup is supported.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Calling forms",
            paragraphs: &[
                "`name` is a string scalar or character vector.",
                "`searchType` is one of `builtin`, `class`, `dir`, `file`, or `var`.",
                "The command form writes the result through normal command output behavior; the function form returns typeID.",
                "One result is returned as a host double scalar. Requesting more than one output rejects before workspace or filesystem lookup; requesting zero outputs still performs the query.",
                "Character rows may have trailing singleton dimensions. Genuine character pages, numeric names, and resident numeric inputs are not text and reject before provider access.",
                "Lookup reads the active workspace, class registry, filesystem, and search path. These session-dependent reads form a fusion boundary and are not scheduled as distributed pure computations.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What data type is typeID?",
            answer: "It is a host double scalar whose value is an integer code. It is not typed-integer storage.",
        },
        BuiltinDocumentationFaq {
            question: "Do wildcard names work?",
            answer: "No. Names containing wildcard characters do not perform a wildcard search and return 0.",
        },
        BuiltinDocumentationFaq {
            question: "Does RunMat support every MATLAB environment-backed code?",
            answer: "Filesystem MEX, P-code, Simulink-extension, builtin, folder, file, and RunMat class discovery are represented. JVM discovery, loaded Simulink state, and arbitrary web URL lookup need broader environment infrastructure.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "MathWorks exist documentation",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/exist.html"),
        },
        BuiltinDocumentationLink {
            label: "isfile",
            target: BuiltinDocumentationLinkTarget::Builtin("isfile"),
        },
        BuiltinDocumentationLink {
            label: "isfolder",
            target: BuiltinDocumentationLinkTarget::Builtin("isfolder"),
        },
        BuiltinDocumentationLink {
            label: "which",
            target: BuiltinDocumentationLinkTarget::Builtin("which"),
        },
        BuiltinDocumentationLink {
            label: "Implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/exist/mod.rs"),
        },
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Lookup, precedence, compatibility, and admission",
                location: "builtins::io::repl_fs::exist::tests",
            }
        ],
        notes: &[
            "Filesystem and symbol results depend on the active session; the examples use isolated fixtures and verify numeric codes rather than ambient machine state."
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

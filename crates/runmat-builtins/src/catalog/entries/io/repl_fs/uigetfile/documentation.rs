use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("uigetfile"),
    slug: Some("uigetfile"),
    summary: "Open a host file-selection dialog and return the selected file name and path.",
    description: "`uigetfile` requests an open-file selection from the selected RunMat session host and returns MATLAB-compatible output values. Interactive hosts can implement the typed hook to show a native or browser picker; noninteractive sessions return the standard cancellation values.",
    keywords: &[
        "uigetfile",
        "file picker",
        "file dialog",
        "open file",
        "filesystem",
        "ui"
    ],
    related: &[
        "fullfile",
        "readtable"
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Calling forms and results",
            paragraphs: &[
                "`file = uigetfile()` uses an all-files filter.",
                "`uigetfile(filter)` accepts a character vector, string scalar, semicolon-delimited pattern list, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
                "`uigetfile(filter, title, defaultName)` forwards the dialog title and initial file or directory path to the selected session host.",
                "`uigetfile(..., 'MultiSelect', 'on')` allows multiple selected files. When multiple files are selected, `file` is a cell array of character vectors. A single selected file is returned as a character vector.",
                "The selected `path` output is the containing directory with a trailing path separator. The `index` output is the one-based selected filter index.",
                "Cancellation returns numeric zero for `file`, `path`, and `index`, matching MATLAB. This is also the behavior when the selected session host does not implement an open-file picker.",
                "Host text and filter cells are validated before any dialog request. Genuine character or cell pages reject; trailing singleton dimensions retain their effective row/matrix shape. Numeric resident values are not text and do not trigger a gather.",
                "Requesting more than three outputs rejects before opening the dialog. Requesting zero outputs still performs the host interaction and validates its result.",
                "MultiSelect accepts on/off in MATLAB compatibility mode. In RunMat mode it additionally accepts logical values, finite double scalars, all eight fixed-width integer scalars, and true/false/yes/no text aliases. Numeric controls use exact zero/nonzero decoding; these extensions reject before host access in MATLAB mode.",
            ]
        },
        BuiltinDocumentationSection {
            heading: "Host integration",
            paragraphs: &[
                "No. File-selection dialogs are host/UI operations and always execute outside GPU residency. The provider hook returns paths in the active filesystem namespace, so browser hosts that import user-selected files should map them into the mounted provider namespace before returning the path.",
                "Host interaction is an effectful callback and fusion boundary, not GPU computation or a distributed worker task. Interactive examples use typed DesktopHost fixtures; headless cancellation does not verify an actual picker.",
            ]
        }
    ],
    examples: super::examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What happens in noninteractive or headless RunMat hosts?",
            answer: "The runtime returns numeric cancellation zeros when the selected session host has no picker. An interactive session host implements select_file_open."
        },
        BuiltinDocumentationFaq {
            question: "How does a browser host provide a real file picker?",
            answer: "The host must connect its interactive picker to the runtime's host-dialog service and return paths in the active filesystem namespace. Filesystem storage alone does not install a picker. Native host-dialog requests and selections are typed; browser applications need the corresponding host bridge."
        },
        BuiltinDocumentationFaq {
            question: "Does `uigetfile` read the selected file?",
            answer: "No. It only returns the selected file name and directory. Use `fullfile(path, file)` with `readtable`, `fopen`, `load`, or another file-reading builtin to consume the file."
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "fullfile",
            target: BuiltinDocumentationLinkTarget::Builtin("fullfile")
        },
        BuiltinDocumentationLink {
            label: "readtable",
            target: BuiltinDocumentationLinkTarget::Builtin("readtable")
        },
        BuiltinDocumentationLink {
            label: "Implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/uigetfile/mod.rs")
        }
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Dialog requests, outputs, controls and admission",
                location: "builtins::io::repl_fs::uigetfile::tests"
            }
        ],
        notes: &[
            "Host bridge and interaction receipt qualification is separate from noninteractive cancellation tests."
        ]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

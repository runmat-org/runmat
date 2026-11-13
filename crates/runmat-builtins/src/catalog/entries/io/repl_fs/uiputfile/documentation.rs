use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("uiputfile"),
    slug: Some("uiputfile"),
    summary: "Open a host save-file dialog and return the selected file name and path.",
    description: "`uiputfile` requests a save-file target from the selected RunMat session host and returns MATLAB-compatible output values. Interactive hosts can implement the typed hook to show a native or browser save picker; noninteractive sessions return the standard cancellation values.",
    keywords: &[
        "uiputfile",
        "file picker",
        "file dialog",
        "save file",
        "filesystem",
        "ui"
    ],
    related: &[
        "fullfile",
        "save"
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Calling forms and results",
            paragraphs: &[
                "`file = uiputfile()` uses an all-files filter.",
                "`uiputfile(filter)` accepts a character vector, string scalar, semicolon-delimited pattern list, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
                "`uiputfile(filter, title, defaultName)` forwards the dialog title and initial file or directory path to the selected session host.",
                "The selected `path` output is the containing directory with a trailing path separator. The `index` output is a one-based integer-valued double, not a native integer-class result.",
                "Inputs are textual. Host or provider-resident numeric values are rejected before the save-file request is made.",
                "Cancellation returns numeric zero for `file`, `path`, and `index`, matching MATLAB. This is also the behavior when the selected session host does not implement a save-file picker.",
                "Host text and filter cells are validated before any dialog request. Genuine character or cell pages reject; trailing singleton dimensions retain their effective row/matrix shape. Numeric resident values are not text and do not trigger a gather.",
                "Requesting more than three outputs rejects before opening the dialog. Requesting zero outputs still performs the host interaction and validates its result.",
            ]
        },
        BuiltinDocumentationSection {
            heading: "Host integration",
            paragraphs: &[
                "No. Save-file dialogs are host/UI operations and always execute outside GPU residency. The provider hook returns a path in the active filesystem namespace, so browser hosts should map save targets into the mounted provider namespace before returning the path.",
                "Host interaction is an effectful callback and fusion boundary, not GPU computation or a distributed worker task. Interactive examples use typed DesktopHost fixtures; headless cancellation does not verify an actual picker.",
            ]
        }
    ],
    examples: super::examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What happens in noninteractive or headless RunMat hosts?",
            answer: "The runtime returns numeric cancellation zeros when the selected session host has no picker. An interactive session host implements select_file_save."
        },
        BuiltinDocumentationFaq {
            question: "How does a browser host provide a real save picker?",
            answer: "The host must connect its interactive picker to the runtime's host-dialog service and return paths in the active filesystem namespace. Filesystem storage alone does not install a picker. Native host-dialog requests and selections are typed; browser applications need the corresponding host bridge."
        },
        BuiltinDocumentationFaq {
            question: "Does `uiputfile` write the selected file?",
            answer: "No. It only returns the selected file name and directory. Use `fullfile(path, file)` with `save`, `fopen`, `writematrix`, or another file-writing builtin to create the file."
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "fullfile",
            target: BuiltinDocumentationLinkTarget::Builtin("fullfile")
        },
        BuiltinDocumentationLink {
            label: "save",
            target: BuiltinDocumentationLinkTarget::Builtin("save")
        },
        BuiltinDocumentationLink {
            label: "Implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/uiputfile/mod.rs")
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
                location: "builtins::io::repl_fs::uiputfile::tests"
            }
        ],
        notes: &[
            "Host bridge and interaction receipt qualification is separate from noninteractive cancellation tests."
        ]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

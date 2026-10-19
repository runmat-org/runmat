use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("uigetdir"),
    slug: Some("uigetdir"),
    summary: "Open a host folder-selection dialog and return the selected folder path.",
    description: "`uigetdir` requests a folder selection from the selected RunMat session host and returns MATLAB-compatible output values. Interactive hosts can implement the typed hook to show a native or browser folder picker; noninteractive sessions return the standard cancellation value.",
    keywords: &[
        "uigetdir",
        "folder dialog",
        "directory picker",
        "filesystem"
    ],
    related: &[
        "uigetfile",
        "uiputfile",
        "dir"
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Calling forms and results",
            paragraphs: &[
                "folder = uigetdir(): Open a folder-selection dialog using the provider's default starting folder.",
                "folder = uigetdir(startPath): Open a folder-selection dialog with an initial folder.",
                "folder = uigetdir(startPath, title): Open a folder-selection dialog with an initial folder and title.",
                "`folder` is a character vector containing the selected folder path.",
                "`folder` is `0` when the user cancels or the active noninteractive provider has no folder picker.",
                "`uigetdir` does not create, read, or list folders itself; it only returns the selected folder path.",
                "RunMat validates malformed provider selections and reports deterministic `RunMat:uigetdir:*` errors.",
                "Starting paths and titles must be scalar text. Character rows with trailing singleton dimensions are accepted; genuine pages and numeric or resident numeric values reject before provider access.",
                "An excessive output count rejects before opening the dialog. A zero-output call still requests a selection, and malformed selections or provider failures still report errors.",
            ]
        },
        BuiltinDocumentationSection {
            heading: "Host integration",
            paragraphs: &[
                "Native and browser sessions use their selected RuntimeHostService. Interactive selections require that exact session host to implement the directory-picker hook; an absent hook follows the documented cancellation behavior rather than fabricating a selected path.",
                "Dialog requests are host callbacks and fusion boundaries. They are not GPU operations or distributed worker computations. The documented examples require the DesktopHost interaction fixture; a command-line-only run is not evidence that an interactive picker worked.",
            ]
        },
    ],
    examples: super::examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
    ],
    media: &[
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "Implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/uigetdir/mod.rs")
        }
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Admission, host requests, selections and cancellation",
                location: "builtins::io::repl_fs::uigetdir::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "VM directory dialog",
                location: "crates/runmat-vm/tests/io_uigetdir.rs"
            },
        ],
        notes: &[
        ]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

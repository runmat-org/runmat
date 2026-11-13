use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation {
    authority:BuiltinDocumentationAuthority::Catalog,
    title:Some("open"),
    slug:Some("open"),
    summary:"Open a file or variable using MATLAB-compatible dispatch.",
    description:"`open(name)` resolves a workspace variable or file using RunMat's search path. MAT-files return a struct; RunMat figure archives dispatch through `openfig` and return a figure handle; custom `open<extension>` handlers receive the resolved path. Editor, viewer, and variable requests are delivered to the active application host. These routes report an error when no supporting host is installed.",
    keywords:&[
        "open",
        "file",
        "MAT-file",
        "viewer",
        "extension handler"
    ],
    related:&[
        "load",
        "which",
        "exist",
        "run"
    ],
    sections:&[
        BuiltinDocumentationSection{
            heading:"Dispatch and host interaction",
            paragraphs:&[
                "`open name` and `A = open(name)` accept a string scalar or row character vector.",
                "Workspace variable names request the active host's variable editor and return an empty 0-by-0 array after the host accepts the request.",
                "Names without an extension are searched on the current RunMat path using MATLAB-like file extensions.",
                "Names with an explicit extension are first searched exactly on the current RunMat path so custom extensions such as `.log` can dispatch to `openlog`.",
                "MAT-files are decoded through the same implementation as `load` and return a scalar struct with one field per variable.",
                "`.fig` files dispatch to `openfig` and return its numeric figure handle. RunMat figure archives are not decoded as MAT-file containers.",
                "For non-MAT files, RunMat looks for a semantic handler named `open<extension>` and passes the resolved file path as a string.",
                "Known text, script, project/model, document, toolbox/install, executable, and spreadsheet extensions request the active host's editor or viewer. They return an empty 0-by-0 array after host acceptance, not a no-op success.",
                "Requesting more than one output raises `RunMat:open:TooManyOutputs`.",
                "Unsupported file types without a handler raise `RunMat:open:UnsupportedFileType`.",
                "Editor and viewer routes raise `RunMat:host:DocumentInteractionUnavailable` when the application host has no document service. A host may reject file types it cannot display.",
                "The name must already be host text. Numeric and GPU-resident names reject without downloads. Character rows may have trailing singleton dimensions, but genuine character pages are not scalar text.",
                "A custom handler returning no outputs still handles the request; RunMat does not fall through to a viewer afterward. A handler can return any supported value class or residency, and its own effects are preserved.",
            ]
        },
        BuiltinDocumentationSection{
            heading:"Execution boundaries",
            paragraphs:&[
                "File resolution and loading use the active filesystem provider. Native, compiled, and browser execution share runtime dispatch; editor routes additionally require the application host service.",
                "This is an effectful fusion boundary, not a numeric GPU kernel or distributed operation. Custom handlers execute their own operations and are not restricted to host-only output."
            ]
        }
    ],
    examples:super::examples::EXAMPLES,
    example_exemption:None,
    faqs:&[
        BuiltinDocumentationFaq{
            question:"Does `open` launch a desktop editor or external application?",
            answer:"Editor and viewer requests go through the active application's document service. The service decides how to display the resolved path or workspace variable. A standalone runtime without that service reports `RunMat:host:DocumentInteractionUnavailable`; MAT loading and custom extension handlers do not require an editor."
        },
        BuiltinDocumentationFaq{
            question:"How is `open('file.mat')` different from `load('file.mat')`?",
            answer:"For MAT-files, `open` returns the same struct-style data that `load` returns when called with one output."
        },
        BuiltinDocumentationFaq{
            question:"Can user code customize how an extension opens?",
            answer:"Yes. Define a function named `open` followed by the lowercase extension, for example `openabc` for `.abc` files. RunMat calls that handler with the resolved filename."
        }
    ],
    links:&[
        BuiltinDocumentationLink {
            label: "Implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/open/mod.rs"),
        },
        BuiltinDocumentationLink{
            label:"load",
            target:BuiltinDocumentationLinkTarget::Builtin("load")
        },
        BuiltinDocumentationLink{
            label:"which",
            target:BuiltinDocumentationLinkTarget::Builtin("which")
        },
        BuiltinDocumentationLink{
            label:"exist",
            target:BuiltinDocumentationLinkTarget::Builtin("exist")
        },
        BuiltinDocumentationLink{
            label:"run",
            target:BuiltinDocumentationLinkTarget::Builtin("run")
        }
    ],
    media:&[
    ],
    evidence:BuiltinDocumentationEvidence{
        implementation:&[
        ],
        verification:&[
            BuiltinEvidenceReference{
                kind:BuiltinEvidenceKind::UnitTest,
                label:"Dispatch, host requests, output policy, and handlers",
                location:"builtins::io::repl_fs::open::tests"
            }
        ],
        notes:&[
        ]
    },
    introduced:None,
    status:Some(BuiltinDocumentationStatus::Stable),
};

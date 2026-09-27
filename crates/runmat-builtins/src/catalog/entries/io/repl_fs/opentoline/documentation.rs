use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation {
    authority:BuiltinDocumentationAuthority::Catalog,
    title:Some("opentoline"),
    slug:Some("opentoline"),
    summary:"Open a file to a requested editor line in MATLAB-compatible code.",
    description:"opentoline resolves a file and sends a typed navigation request to the active application host. The request contains a one-based line, column, and caret or selection intent. Hosts without an editor service report an error rather than silently ignoring the request.",
    keywords:&[
        "opentoline",
        "open",
        "editor",
        "file",
        "line",
        "column"
    ],
    related:&[
        "open",
        "which",
        "exist"
    ],
    sections:&[
        BuiltinDocumentationSection{
            heading:"Positions and host navigation",
            paragraphs:&[
                "`opentoline(filename,line)` accepts a string scalar, row character vector, or scalar string array filename plus a positive integer line number.",
                "`opentoline(filename,line,column)` also validates a positive integer column number.",
                "Native typed-integer line and column spellings are RunMat extensions because the public contract establishes structural positions without enumerating native numeric classes.",
                "Explicit `gpuArray` positions are a separate RunMat extension; automatically resident compatible scalars still gather through their owning provider in MATLAB-compatible mode.",
                "opentoline(filename,line,column,'select') asks the host to select at the requested location instead of placing only the caret.",
                "Command-style numeric line and column arguments passed as text, such as `opentoline file.m 10 4`, are accepted when they parse as positive integers.",
                "File names are resolved using the RunMat search path and the same general MATLAB-style file extensions used by `open`.",
                "A missing file raises `RunMat:opentoline:FileNotFound`.",
                "Non-scalar, nonfinite, fractional, zero, or negative line and column values raise `RunMat:opentoline:InvalidPosition`.",
                "Requesting output arguments raises `RunMat:opentoline:TooManyOutputs`.",
                "The host must accept the navigation request before the call succeeds. Missing editor support raises RunMat:host:DocumentInteractionUnavailable.",
                "Filename, option, host positions, and requested output count are validated before resident downloads or file lookup. Text arguments do not trigger GPU downloads. Character rows can carry trailing singleton dimensions; genuine pages are not scalar text.",
                "Native integer positions are decoded in their integer class without a floating-point round trip and must fit the host address-sized positive index range.",
            ]
        },
        BuiltinDocumentationSection{
            heading:"Execution boundaries",
            paragraphs:&[
                "Editor navigation is a host-side, effectful fusion boundary. It does not run as a numeric GPU kernel or distributed worker operation. Admitted resident position scalars are downloaded through their owning provider; explicit residency and integer compatibility gates remain independent.",
                "Native, JIT, AOT, and browser execution use the same runtime binding. Each embedding must install its own document host service; successful compilation does not establish that an editor is available."
            ]
        }
    ],
    examples:super::examples::EXAMPLES,
    example_exemption:None,
    faqs:&[
        BuiltinDocumentationFaq{
            question:"Does opentoline work in every host?",
            answer:"It requires an application host that implements document navigation. Desktop and browser applications can supply that service; the runtime itself does not launch an editor or pretend to navigate without one."
        },
        BuiltinDocumentationFaq{
            question:"Why is this an I/O operation rather than a plotting function?",
            answer:"It resolves a source file and requests editor navigation. It does not create or modify a figure."
        }
    ],
    links:&[
        BuiltinDocumentationLink {
            label: "Implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/opentoline/mod.rs"),
        },
        BuiltinDocumentationLink{
            label:"open",
            target:BuiltinDocumentationLinkTarget::Builtin("open")
        },
        BuiltinDocumentationLink{
            label:"which",
            target:BuiltinDocumentationLinkTarget::Builtin("which")
        },
        BuiltinDocumentationLink{
            label:"exist",
            target:BuiltinDocumentationLinkTarget::Builtin("exist")
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
                label:"Typed positions, compatibility, admission and host requests",
                location:"builtins::io::repl_fs::opentoline::tests"
            },
            BuiltinEvidenceReference{
                kind:BuiltinEvidenceKind::IntegrationTest,
                label:"VM host navigation",
                location:"basics::opentoline_dispatches_editor_navigation_request"
            }
        ],
        notes:&[
        ]
    },
    introduced:None,
    status:Some(BuiltinDocumentationStatus::Stable),
};

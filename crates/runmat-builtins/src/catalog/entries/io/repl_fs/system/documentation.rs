use crate::*;

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "status-and-output",
    title: "Capture a host command's output",
    program: "[status, output] = system('echo runmat');\nassert(status == 0);\nassert(contains(output, 'runmat'));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(status == 0); assert(contains(output, 'runmat'));",
    },
}];

pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("system"),
    slug: Some("system"),
    summary: "Run a native host command and return its exit status and complete output.",
    description: "`system(command)` runs a shell command on a native host. With two outputs it returns a double status and a character row containing complete standard output followed by complete standard error. An optional `'-echo'` also sends the collected text to the RunMat console.",
    keywords: &["system", "shell", "command", "process"],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &[
                "The command and optional `'-echo'` control must be host string scalars or character rows. Numeric and GPU-resident controls reject before child creation. Zero, one, and two requested outputs are supported; zero outputs writes collected command text to the console.",
                "On native Windows the command uses `cmd /C`; on Unix-like hosts it uses `/bin/sh -c`. The active filesystem provider must issue a real native working-directory grant. Virtual filesystem paths are not used as operating-system paths.",
                "The child inherits the RunMat session's environment, including changes made with `setenv`. A frozen session exposes only its installed environment; neither session changes nor child execution modify the embedding process's environment.",
                "Both output pipes are drained concurrently without truncation. The result groups stdout before stderr; it does not preserve cross-stream interleaving. `'-echo'` writes that same collected result to the RunMat console after the command finishes.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Availability and limits",
            paragraphs: &[
                "A browser without a process host reports that process execution is unavailable after checking the command and option. It does not return a fabricated exit status.",
                "Complete process output is currently unbounded. If the installed execution profile imposes a finite host-memory budget, the call rejects before starting a child until an enforceable output allocation bound is available.",
                "Cancellation or abandonment terminates the owned child process tree. Operating-system spawn, output, and wait failures retain their causes; a nonzero child exit remains a returned status rather than an exception.",
                "Core installs the native process service. An embedding that invokes this binding directly must install the same typed service; an absent service reports a capability error instead of spawning through an ambient fallback.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Is the order of stdout and stderr preserved?",
            answer: "The returned text contains all stdout first and all stderr second. The order of writes between the two pipes cannot be reconstructed from that result.",
        },
        BuiltinDocumentationFaq {
            question: "Can a command use a virtual project folder as its native working folder?",
            answer: "No. The selected filesystem must explicitly issue a native directory grant; a virtual path is not silently translated to a host path.",
        },
    ],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB system reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/system.html",
        ),
    }, BuiltinDocumentationLink {
        label: "MATLAB ispc reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/ispc.html",
        ),
    }, BuiltinDocumentationLink {
        label: "MATLAB ismac reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/ismac.html",
        ),
    }, BuiltinDocumentationLink {
        label: "MATLAB isunix reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/isunix.html",
        ),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime admission and output adaptation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/system.rs"),
            },
            BuiltinDocumentationLink {
                label: "Complete process output owner",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-process-host/src/command.rs"),
            },
            BuiltinDocumentationLink {
                label: "Native working-directory grant",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-filesystem/src/native_path.rs"),
            },
            BuiltinDocumentationLink {
                label: "Core process service and shared host driver",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-core/src/session/process_host.rs"),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Complete dual-pipe output and owned-child cancellation fixtures",
                location: "runmat_process_host::command::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "System output, invalid control, finite budget, and virtual directory fixtures",
                location: "builtins::io::repl_fs::compat::system::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Ordinary Core execution without a Tokio caller",
                location: "runmat_core::tests::system_executes_from_ordinary_core_without_a_tokio_caller",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

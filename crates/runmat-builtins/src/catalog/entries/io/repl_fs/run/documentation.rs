use super::examples::EXAMPLES;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Script resolution and workspace behavior",
        paragraphs: &[
            "`run('script')` resolves `script.p` before `script.m` when no extension is supplied. Bare `.p` and `.m` filenames also search the active MATLAB path. `run('folder/script.m')` executes the file at that path.",
            "RunMat-generated `.p` files are decoded before parsing. Proprietary or non-RunMat `.p` files report `RunMat:pcode:InvalidPcode`.",
            "Command syntax is supported: `run script_name` is equivalent to `run('script_name')`.",
            "The target file is executed as a script, not as a function. Assignments mutate the caller workspace and local functions in the script remain available to that script.",
            "`mfilename` inside the executed script reports the executed file name and full path, matching the script source context.",
            "`run` has no outputs. Requesting an output raises `RunMat:run:TooManyOutputs` before executing the file.",
            "If the script throws an error, the error is catchable by the caller and assignments performed before that error remain in the caller workspace.",
            "Names that may be introduced by a script are tracked as dynamic workspace bindings. Bare reads of names the script did not assign raise `RunMat:UndefinedVariable` at runtime unless the name is a builtin constant; call syntax such as `name(...)` checks the caller workspace first, then falls back to normal function or builtin resolution.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution boundaries",
        paragraphs: &[
            "The selected workspace execution service runs the script in native-hosted and browser sessions. A direct call without an execution context reports RunMat:run:RequiresVm. Compiled execution uses the same source and workspace service boundary.",
            "The script path must be scalar text. Numeric and GPU-resident numeric paths reject before provider access. Script resolution is host-owned and forms a fusion boundary; operations inside the script retain their own acceleration, capability, and ownership rules.",
            "A script can perform arbitrary supported effects, call functions, and create or replace bindings. Its contents are not known from the path argument's static type, so scheduling and worker admission must retain that uncertainty.",
        ],
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `run` call functions?",
        answer: "No. `run` executes script files. Function files should be called by their function name.",
    },
    BuiltinDocumentationFaq {
        question: "Can `run` return a value?",
        answer: "No. MATLAB's `run` executes for side effects in the caller workspace and does not return outputs.",
    },
    BuiltinDocumentationFaq {
        question: "Where does `run` search?",
        answer: "RunMat first resolves the provided file or relative path, then searches the active MATLAB path. It checks `.p` before `.m` when the supplied name has no extension, and a sibling `.p` takes precedence when the resolved source path names `.m`.",
    },
    BuiltinDocumentationFaq {
        question: "What happens if the script creates variables?",
        answer: "Assignments made by the script are visible in the caller workspace after `run`. Subsequent call syntax uses MATLAB variable precedence, so a script-created array can be indexed with `x(...)` and can shadow a builtin name.",
    },
    BuiltinDocumentationFaq {
        question: "What happens if the script fails?",
        answer: "The error propagates to the caller and can be caught with `try`/`catch`. Assignments made before the error remain visible in the caller workspace, matching script execution semantics.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "addpath",
        target: BuiltinDocumentationLinkTarget::Builtin("addpath"),
    },
    BuiltinDocumentationLink {
        label: "path",
        target: BuiltinDocumentationLinkTarget::Builtin("path"),
    },
    BuiltinDocumentationLink {
        label: "which",
        target: BuiltinDocumentationLinkTarget::Builtin("which"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/run/mod.rs"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("run"),
    slug: Some("run"),
    summary: "Execute a MATLAB script file in the caller workspace.",
    description: "`run` locates a `.p` or `.m` script by file name, relative path, absolute path, or active MATLAB search path entry, then executes that script in the caller workspace. Variables assigned by the script are visible to following statements in the caller.",
    keywords: &[
        "run",
        "script",
        "matlab script",
        "workspace",
        "path",
    ],
    related: &[
        "addpath",
        "path",
        "which",
    ],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Source admission and resolution",
                location: "builtins::io::repl_fs::run::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "run_executes_script_file_in_current_workspace",
                location: "runmat-core::tests::run_executes_script_file_in_current_workspace",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "run_exposes_script_variables_to_following_call_syntax",
                location: "runmat-core::tests::run_exposes_script_variables_to_following_call_syntax",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "run_script_variables_shadow_builtin_constants",
                location: "runmat-core::tests::run_script_variables_shadow_builtin_constants",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "run_replays_script_stdout_to_request_streams",
                location: "runmat-core::tests::run_replays_script_stdout_to_request_streams",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "run_command_syntax_resolves_scripts_on_search_path",
                location: "runmat-core::tests::run_command_syntax_resolves_scripts_on_search_path",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "run_preserves_script_source_context",
                location: "runmat-core::tests::run_preserves_script_source_context",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "run_missing_file_and_output_errors_are_catchable",
                location: "runmat-core::tests::run_missing_file_and_output_errors_are_catchable",
            },
        ],
        notes: &[
            "Executable examples supply isolated script fixtures. The legacy GPU text gathering claim was incorrect: GPU handles represent numeric values, which are not script paths.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

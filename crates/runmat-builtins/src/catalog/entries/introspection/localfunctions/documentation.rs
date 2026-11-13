use crate::*;

const LOCAL_FILE: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "named-locals" },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File {
        relative_path: "probe_locals.m",
        content: BuiltinFixtureContent::Utf8("function handles = probe_locals()\nhandles = localfunctions();\nend\nfunction value = child_local()\nvalue = 7;\nend\n"),
    }],
});

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("localfunctions"), slug: Some("localfunctions"),
    summary: "Return handles to named local functions in the current source file.",
    description: "`localfunctions()` returns a 1-by-N cell row of bound function handles in the current source file. With no source context it returns a 1-by-0 cell array.",
    keywords: &["localfunctions", "function handle", "local function", "source file"],
    related: &["functions", "inputname"],
    sections: &[
        BuiltinDocumentationSection { heading: "Current source", paragraphs: &[
            "The runtime queries compiler-owned source-function roles, retaining only `Local` handles and excluding the primary function, nested functions, anonymous functions, methods, and synthetic entrypoints. It does not scan arbitrary files or execute a handle.",
            "The result is a cell row in the runtime catalog's current order. MATLAB does not promise handle order; do not rely on a particular ordering across source/catalog changes.",
            "Input arguments are not accepted and raise `RunMat:TooManyInputs`. The current runtime does not separately identify script, function-file, nested, and command-line contexts with MATLAB's full semantics.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Native and browser runtimes use the same source-function catalog. Handles remain bound to their source-local function identities and are not provider payloads. Transfer to a worker requires that source/function identity to be available there.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "file-local", title: "Find a named local function",
        program: "handles = probe_locals();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: LOCAL_FILE,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(iscell(handles)); found = false; for i = 1:numel(handles); info = functions(handles{i}); if strcmp(info.function,'child_local'); found = true; end; end; assert(found);",
        },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does it return an ordered list of names?",
        answer: "No. It returns a cell row of bound function handles; source/catalog order is not a portable sorting guarantee.",
    }],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB localfunctions reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/localfunctions.html"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime source-catalog lookup",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/localfunctions.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Source-local bound handle test",
            location: "builtins::introspection::localfunctions::tests",
        }, BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Compiled local handle invocation",
            location: "crates/runmat-vm/tests/functions.rs::localfunctions_returns_callable_local_function_handles",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

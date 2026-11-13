use super::super::documentation_support::*;
use crate::*;

pub(super) const PYENV: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pyenv"), slug: Some("pyenv"),
    summary: "Inspect or configure the session Python runtime.",
    description: "`environment = pyenv()` reports the selected Python installation and status; `pyenv('Version', value, 'ExecutionMode', mode)` configures it before startup.",
    keywords: &["pyenv", "python", "environment", "interpreter", "version", "execution mode"],
    related: &["pyargs", "pyrun", "pyrunfile", "terminate"],
    sections: &[
        BuiltinDocumentationSection { heading: "Configuration", paragraphs: &["Supported name-value options are `Version` (a major.minor version or executable path) and `ExecutionMode` (`InProcess` or `OutOfProcess`). Configuration is rejected once the persistent interpreter has started. Status inspection does not start the persistent session, but discovery can launch a short-lived interpreter probe; configuration also discovers a matching installation."] },
        BuiltinDocumentationSection { heading: "Environment result", paragraphs: &["The py.PythonEnvironment value object reports Version, Executable, Library, Home, Status, ExecutionMode, ProcessID, and ProcessName. `terminate` is available only for an out-of-process environment. A browser does not embed CPython and reports unsupported Python interoperability."] },
    ],
    examples: &[BuiltinExample { id: "inspect-python", title: "Inspect the selected Python environment", program: "env = pyenv();\nenv.Status", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeForeignRuntime, fixture: IN_PROCESS_FIXTURE, requirements: PYTHON_HOST, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(env), 'py.PythonEnvironment')); assert(strcmp(env.Status, 'NotLoaded'));" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can I change ExecutionMode after Python starts?", answer: "No. Select the mode before startup. An in-process CPython interpreter cannot be unloaded safely." }],
    links: &[PYTHON_LINK], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Status/configure routing", location: "builtins::interop::python::tests::pyenv_status_and_configure_are_distinct_foreign_symbols" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

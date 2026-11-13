use super::super::documentation_support::*;
use crate::*;

pub(super) const TERMINATE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("terminate"), slug: Some("terminate"),
    summary: "Terminate an out-of-process Python environment.",
    description: "`environment = terminate(environment)` stops a session's out-of-process Python host and returns the updated environment status.",
    keywords: &["terminate", "python", "environment", "process", "out of process"],
    related: &["pyenv", "pyrun", "pyrunfile"],
    sections: &[BuiltinDocumentationSection { heading: "Ownership and availability", paragraphs: &["Only an out-of-process Python environment can be terminated; an in-process CPython interpreter cannot be safely unloaded. Outstanding Python foreign handles become invalid after host restart. The environment input must be a PythonEnvironment value returned by pyenv. Browser sessions have no embedded CPython host."] }],
    examples: &[BuiltinExample { id: "terminate-isolated", title: "Stop an out-of-process environment", program: "env = pyenv();\nenv = terminate(env);\nenv.Status", display_output: Some("Terminated"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeForeignRuntime, fixture: OUT_OF_PROCESS_FIXTURE, requirements: PYTHON_HOST, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(env), 'py.PythonEnvironment')); assert(strcmp(env.Status, 'Terminated'));" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can `terminate` unload in-process Python?", answer: "No. Use OutOfProcess mode when the interpreter must be restartable." }],
    links: &[PYTHON_LINK], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original environment forwarding", location: "builtins::interop::python::tests::terminate_forwards_original_environment_to_foreign_owner" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

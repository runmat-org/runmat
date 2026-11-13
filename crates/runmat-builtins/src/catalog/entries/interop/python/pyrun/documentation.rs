use super::super::documentation_support::*;
use crate::*;

pub(super) const PYRUN: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pyrun"), slug: Some("pyrun"),
    summary: "Execute Python code in the session's persistent Python workspace.",
    description: "`pyrun(code, name, value, ...)` executes without requesting outputs; `result = pyrun(code, outputNames, name, value, ...)` returns named Python values.",
    keywords: &["pyrun", "python", "code", "workspace", "named inputs", "named outputs"],
    related: &["pyenv", "pyrunfile", "pyargs"],
    sections: &[BuiltinDocumentationSection { heading: "Execution", paragraphs: &["Source runs in a persistent workspace for this Python session. Name-value inputs are converted at the foreign boundary; requested output names may be text, a string array, or a cell of text. Python exceptions retain their type, cause, and traceback. Returned opaque Python values remain session-owned foreign objects; compatible NumPy arrays can borrow read-only host buffers in-process and copy when mutation, layout, device residency, or process isolation requires it."] }],
    examples: &[BuiltinExample { id: "named-output", title: "Exchange named values with Python", program: "answer = pyrun('import marker; answer = marker.add2(input_value)', 'answer', 'input_value', int32(40));\nanswer", display_output: Some("answer = 42"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeForeignRuntime, fixture: IN_PROCESS_FIXTURE, requirements: PYTHON_HOST, verification: BuiltinExampleVerification::Assertions { source: "assert(answer == 42);" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does `pyrun` share variables between calls?", answer: "Yes. It uses the session's persistent Python workspace; `pyrunfile` has a separate script workspace." }],
    links: &[PYTHON_LINK], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Named input/output routing", location: "builtins::interop::python::tests::run_and_runfile_preserve_named_input_and_output_order_at_foreign_boundary" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

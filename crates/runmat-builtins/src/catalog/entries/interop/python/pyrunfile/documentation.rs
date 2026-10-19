use super::super::documentation_support::*;
use crate::*;

pub(super) const PYRUNFILE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pyrunfile"), slug: Some("pyrunfile"),
    summary: "Execute a Python file in an isolated script workspace.",
    description: "`pyrunfile(file, ...)` executes a Python script. A requested output requires outputNames after file; name-value inputs and an optional `Arguments` cell of text follow.",
    keywords: &["pyrunfile", "python", "file", "script", "arguments"],
    related: &["pyrun", "pyenv", "pyargs"],
    sections: &[BuiltinDocumentationSection { heading: "Inputs and execution", paragraphs: &["The file path is text. Script definitions do not enter the persistent `pyrun` workspace. Named inputs are passed in pairs; `Arguments` supplies a cell array of command-line argument text. The script file is read by the Python host and must be available in that host's filesystem. Out-of-process calls suspend while the child executes; callbacks and returned foreign values remain tied to the RunMat session."] }],
    examples: &[BuiltinExample { id: "script-output", title: "Read a named output from a Python file", program: "answer = pyrunfile('answer.py', 'answer', 'input_value', int32(40));\nanswer", display_output: Some("answer = 42"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeForeignRuntime, fixture: IN_PROCESS_FIXTURE, requirements: PYTHON_HOST, verification: BuiltinExampleVerification::Assertions { source: "assert(answer == 42);" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does a file share `pyrun` variables?", answer: "No. File execution uses a separate script workspace." }],
    links: &[PYTHON_LINK], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "File and named operand routing", location: "builtins::interop::python::tests::run_and_runfile_preserve_named_input_and_output_order_at_foreign_boundary" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

use super::super::documentation_support::*;
use crate::*;

pub(super) const PYARGS: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pyargs"), slug: Some("pyargs"),
    summary: "Create a Python keyword-argument bundle.",
    description: "`bundle = pyargs(name, value, ...)` stores name-value pairs for a later `py.*` call. Bundle construction does not start Python.",
    keywords: &["pyargs", "python", "keyword", "arguments", "bundle"],
    related: &["pyenv", "pyrun", "pyrunfile"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and result", paragraphs: &["Names must be nonempty text and arguments must occur in pairs. Values are retained without Python conversion until the bundle is used by a Python call. The result is a RunMat.PythonArguments value object with `Names` and `Values` cell properties."] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &["Construction is local, works without a CPython service (including in a browser session), and does not create a Python foreign handle. Using the bundle with `py.*` requires an available session Python service."] },
    ],
    examples: &[BuiltinExample { id: "keyword-bundle", title: "Create a keyword bundle", program: "kw = pyargs('rel_tol', 0.00001);\nkw.Names", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(kw), 'RunMat.PythonArguments')); assert(strcmp(kw.Names{1}, 'rel_tol')); assert(kw.Values{1} == 0.00001);" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does `pyargs` start Python?", answer: "No. It only stores keyword names and values; conversion happens when a Python call consumes the bundle." }],
    links: &[PYTHON_LINK], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Portable local bundle", location: "builtins::interop::python::tests::pyargs_constructs_local_bundle_without_foreign_service" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

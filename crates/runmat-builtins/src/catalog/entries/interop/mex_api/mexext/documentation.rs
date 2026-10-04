use super::super::documentation_support::*;
use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mexext"), slug: Some("mexext"),
    summary: "Return the native MEX module filename extension for this target.",
    description: "`extension = mexext()` returns the supported target's module suffix without a leading dot. It does not probe a compiler, install an SDK, compile a source file, or load a MEX module.",
    keywords: &["mexext", "mex", "native extension", "filename suffix", "platform"],
    related: &["mex", "mexcuda"],
    sections: &[
        BuiltinDocumentationSection { heading: "Result", paragraphs: &[
            "The result is a host string scalar such as a supported native target's MEX suffix, with no leading period. It accepts no arguments. The exact suffix is selected by the shared MEX target table rather than guessed from the OS name.",
            "Unsupported targets, including browser/Worker builds, report RunMat:MEX:UnsupportedTarget. The suffix alone does not imply that a compiler or a particular compiled module is available.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "extension", title: "Inspect the native suffix", program: "extension = mexext();\nassert(isstring(extension));\nassert(strlength(extension) > 0);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Native, fixture: BuiltinExampleFixture::None, requirements: NATIVE_TARGET, verification: BuiltinExampleVerification::Assertions { source: "assert(isstring(extension)); assert(strlength(extension) > 0);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `mexext` include a dot?", answer: "No. Add the period when constructing a filename." },
        BuiltinDocumentationFaq { question: "Does it prove MEX compilation is installed?", answer: "No. It reports the target suffix, not compiler, SDK, or loader readiness." },
    ],
    links: &[MEX_LINK, BuiltinDocumentationLink { label: "mex", target: BuiltinDocumentationLinkTarget::Builtin("mex") }, BuiltinDocumentationLink { label: "mexcuda", target: BuiltinDocumentationLinkTarget::Builtin("mexcuda") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Suffix matches target table", location: "builtins::interop::mex::tests::mexext_matches_the_adapter_target_suffix" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("readlines"), slug: Some("readlines"),
    summary: "Read a text file into a column string array.", description: "Read a text file into a column string array.",
    keywords: &["readlines"], related: &["writelines"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Accepts MATLAB text inputs such as string scalars and character row vectors for path, name, or URL arguments where applicable.",
"Runs on the host using the active RunMat filesystem, environment, or HTTP abstraction as appropriate.",
"Reports MATLAB-style runtime errors for unsupported argument shapes or unavailable host capabilities.",
"All textual arguments and controls are admitted before filesystem or network access. Numeric resident values are invalid text and are rejected without accelerator download.",
"Character text allows trailing singleton dimensions but rejects genuine pages. Filesystem operations use the active provider; unsupported host capabilities report errors.",
"Calls are effectful runtime boundaries, not fusion kernels. VM and native/JIT/AOT execution use the same canonical binding.",
"UTF-8 input recognizes LF, CRLF, and CR; one final line terminator does not create an extra row. EmptyLineRule accepts read or skip, with omit and omitempty aliases. Unsupported options and rule values reject before reading."] },
        BuiltinDocumentationSection { heading: "Placement and compatibility", paragraphs: &["Host text and filesystem operation; no accelerator kernel, upload, implicit gather, broadcast, or fusion. Numeric precision controls do not apply.", "No optional feature is required. Unsupported active-filesystem or HTTP capabilities fail rather than bypassing the configured provider."] },
    ],
    examples: &[BuiltinExample {
        id: "example-1", title: "Use isolated filesystem inputs",
        program: "lines = readlines(\"input.txt\", \"EmptyLineRule\", \"skip\");", display_output: Some("The output preserves file content and the documented result shape."),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "text-files" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "input.txt", content: BuiltinFixtureContent::Utf8("alpha\n\nbeta\n") }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(lines, \"string\")); assert(isequal(size(lines), [2 1])); assert(isequal(lines, [\"alpha\"; \"beta\"]));" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime domain", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/lines.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Domain admission and filesystem behavior", location: "builtins::io::filetext::lines::tests" }],
        notes: &["Typed isolated-filesystem examples verify content and result representation; provider, native, browser, and transport qualification is tracked by the exact-revision integrated gates."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

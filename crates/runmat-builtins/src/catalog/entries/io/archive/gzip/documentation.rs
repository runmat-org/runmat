use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("gzip"), slug: Some("gzip"),
    summary: "Compress files using gzip.", description: "Compress files using gzip.",
    keywords: &["gzip"], related: &["gunzip"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Compresses scalar file paths, string arrays, cell arrays of paths, wildcard expansions, and files inside a folder.",
"Writes .gz files beside each input by default or under the supplied output folder.",
"Returns a cell array of generated gzip file paths.",
"All textual arguments and controls are admitted before filesystem or network access. Numeric resident values are invalid text and are rejected without accelerator download.",
"Character text allows trailing singleton dimensions but rejects genuine pages. Filesystem operations use the active provider; unsupported host capabilities report errors.",
"Calls are effectful runtime boundaries, not fusion kernels. VM and native/JIT/AOT execution use the same canonical binding.",
"Wildcard traversal is provider-owned and matches are sorted by path. Folder input enumerates immediate non-directory entries. Each input is fully compressed in host memory before its output write; failures may leave earlier outputs.",
"Compression cooperates with the session's task scheduler. Cancellation during compression prevents that input's output write; outputs already written for earlier inputs remain. Filesystem writes are not transactional."] },
        BuiltinDocumentationSection { heading: "Placement and compatibility", paragraphs: &["Host text and filesystem operation; no accelerator kernel, upload, implicit gather, broadcast, or fusion. Numeric precision controls do not apply.", "No optional feature is required. Unsupported active-filesystem or HTTP capabilities fail rather than bypassing the configured provider."] },
    ],
    examples: &[BuiltinExample {
        id: "example-1", title: "Use isolated filesystem inputs",
        program: "files = gzip(\"input.txt\", \"compressed\");\nrestored = gunzip(files, \"restored\");\ntext = fileread(restored{1});", display_output: Some("The output preserves file content and the documented result shape."),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "text-files" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "input.txt", content: BuiltinFixtureContent::Utf8("alpha\nbeta\n") }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(files)); assert(isequal(size(files), [1 1])); assert(ischar(files{1})); assert(strcmp(text, sprintf(\"alpha\\nbeta\\n\")));" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime domain", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/archive/gzip.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Domain admission and filesystem behavior", location: "builtins::io::archive::gzip::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Codec interoperability and bounded cooperation", location: "builtins::io::archive::gzip::codec::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Selected-session cancellation and output publication", location: "builtins::io::archive::gzip::cooperation_tests" },
        ],
        notes: &["Typed isolated-filesystem examples verify content and result representation; provider, native, browser, and transport qualification is tracked by the exact-revision integrated gates."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

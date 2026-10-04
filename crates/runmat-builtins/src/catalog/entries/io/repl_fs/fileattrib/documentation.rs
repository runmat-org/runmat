use crate::*;

const SAMPLE: &[BuiltinFilesystemEntry] = &[BuiltinFilesystemEntry::File {
    relative_path: "sample.bin",
    content: BuiltinFixtureContent::Bytes(&[1, 2, 3]),
}];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "read-attributes",
    title: "Read available file attributes",
    program: "[status, attributes] = fileattrib('sample.bin');\nassert(status == 1);\nassert(~attributes.directory);\nassert(isnan(attributes.GroupRead));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
        id: BuiltinExampleFixtureId { local_name: "read-attributes" },
        root: BuiltinFilesystemRoot::IsolatedWorkspace,
        entries: SAMPLE,
    }),
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(status == 1); assert(~attributes.directory); assert(isnan(attributes.GroupRead));",
    },
}];

pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fileattrib"),
    slug: Some("fileattrib"),
    summary: "Read file attributes and change the supported writable flag.",
    description: "`fileattrib(path)` returns a status and attribute structure for a host text path. `+w` and `-w` update the provider's read-only flag. Other permission changes are not yet supported. Unavailable permission facts are NaN rather than guessed values.",
    keywords: &["fileattrib", "file", "folder", "attributes", "permissions"],
    related: &["dir", "isfile", "isfolder"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &[
                "A path may be a host string scalar or character row. One or two writable flags may follow; the last flag determines the update. Paths and flags must be known before filesystem access.",
                "The status is 1 when metadata was read and 0 when the selected provider could not find or inspect the path. The second result is an attribute structure or an error-message structure.",
                "The exact filesystem provider and working folder are captured for the call, including the mutation and subsequent metadata read.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Permission limitations",
            paragraphs: &[
                "The current provider supplies file type and a read-only flag, but not user/group/other read or execute permissions or native hidden/archive/system attributes. Fields that cannot be established are NaN; UserWrite reflects the provider read-only flag and is not a complete operating-system access check.",
                "Read-permission mutation (`+r`, `r`, `-r`) reports unsupported permission before any filesystem mutation. Wildcards, user selectors, recursive changes, archive/system mutation, and full MATLAB setter-output forms require a richer provider permission interface.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Why do some permission fields contain NaN?",
            answer: "The active filesystem provider cannot establish those facts. NaN means unavailable, not allowed or denied.",
        },
        BuiltinDocumentationFaq {
            question: "Can I change read permissions?",
            answer: "Not yet. The current provider only supports changing its read-only flag through +w and -w.",
        },
    ],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB fileattrib reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/fileattrib.html"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime binding and attribute preparation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat.rs"),
            },
            BuiltinDocumentationLink {
                label: "Captured filesystem authority",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat/captured_fs_path.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Writable preflight, selected provider, and unknown permission fixtures",
            location: "builtins::io::repl_fs::compat::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

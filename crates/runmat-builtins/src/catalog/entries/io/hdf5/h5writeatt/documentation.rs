use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("h5writeatt"),
    slug: Some("h5writeatt"),
    summary: "Write an HDF5 attribute on a file, group, or dataset.",
    description: "`h5writeatt(filename, location, attribute, data)` creates or replaces an attribute at the root, a group, or a dataset.",
    keywords: &["h5writeatt", "hdf5", "attribute", "dataset", "group", "file", "io"],
    related: &["h5info", "h5disp"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Dense numeric (including exact integer classes), logical, string, and char attributes are supported. Unsupported MATLAB value classes report `RunMat:hdf5:UnsupportedType`.", "The value is admitted before the native file is opened. Existing attribute replacement is not transactional across an operating-system or HDF5 write failure. The host native HDF5 route has no GPU kernel or fusion route; browser execution reports `RunMat:hdf5:UnsupportedTarget`."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Native HDF5 file access is synchronous host work. It terminates fusion and does not launch a GPU kernel. The active native filesystem path authority and HDF5 library must be available; a missing native route reports a typed error rather than falling back to an unrelated ambient path."] },
    ],
    examples: &[BuiltinExample {
        id: "native-h5writeatt",
        title: "h5writeatt with a deterministic native HDF5 file",
        program: "h5write('out.h5', '/A', [1 2]);\nh5writeatt('out.h5', '/A', 'units', 'm/s');\ninfo = h5info('out.h5', '/A');\ninfo.Attributes",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(numel(info.Attributes) == 1); assert(strcmp(info.Attributes{1}.Name, 'units'));" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can this run in a browser?", answer: "The current HDF5 backend requires a native HDF5 library; browser calls return RunMat:hdf5:UnsupportedTarget." },
    ],
    links: &[BuiltinDocumentationLink { label: "h5info", target: BuiltinDocumentationLinkTarget::Builtin("h5info") },
        BuiltinDocumentationLink { label: "h5disp", target: BuiltinDocumentationLinkTarget::Builtin("h5disp") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "mod.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/mod.rs") },
            BuiltinDocumentationLink { label: "geometry.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/geometry.rs") },
            BuiltinDocumentationLink { label: "path.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/path.rs") },
            BuiltinDocumentationLink { label: "write.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/native/write.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Native HDF5 behavior", location: "builtins::io::hdf5::tests::h5writeatt_h5info_and_h5disp_report_metadata" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

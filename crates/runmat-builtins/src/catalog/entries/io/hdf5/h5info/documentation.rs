use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("h5info"),
    slug: Some("h5info"),
    summary: "Return metadata for an HDF5 file, group, or dataset.",
    description: "`h5info(filename)` returns recursive root metadata. `h5info(filename, location)` returns metadata for a selected group or dataset.",
    keywords: &["h5info", "hdf5", "metadata", "dataset", "group", "file", "io"],
    related: &["h5disp", "h5read"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Group metadata contains `Name`, `Attributes`, `Groups`, `Datasets`, `Datatypes`, and `Links`; root metadata also includes `Filename`. Dataset metadata contains `Name`, `Datatype`, `Class`, `Dataspace`, `Size`, and `Attributes`.", "The active native HDF5 route is host file I/O, not GPU or fusion. Browser execution returns `RunMat:hdf5:UnsupportedTarget` until a provider-backed HDF5 codec is installed."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Native HDF5 file access is synchronous host work. It terminates fusion and does not launch a GPU kernel. The active native filesystem path authority and HDF5 library must be available; a missing native route reports a typed error rather than falling back to an unrelated ambient path."] },
    ],
    examples: &[BuiltinExample {
        id: "native-h5info",
        title: "h5info with a deterministic native HDF5 file",
        program: "h5write('measurements.h5', '/group/A', [1 2; 3 4]);\ninfo = h5info('measurements.h5');\ninfo.Groups",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(info.Name, '/')); assert(numel(info.Groups) == 1); assert(strcmp(info.Groups{1}.Name, '/group'));" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can this run in a browser?", answer: "The current HDF5 backend requires a native HDF5 library; browser calls return RunMat:hdf5:UnsupportedTarget." },
    ],
    links: &[BuiltinDocumentationLink { label: "h5disp", target: BuiltinDocumentationLinkTarget::Builtin("h5disp") },
        BuiltinDocumentationLink { label: "h5read", target: BuiltinDocumentationLinkTarget::Builtin("h5read") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "mod.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/mod.rs") },
            BuiltinDocumentationLink { label: "geometry.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/geometry.rs") },
            BuiltinDocumentationLink { label: "path.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/path.rs") },
            BuiltinDocumentationLink { label: "metadata.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/hdf5/native/metadata.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Native HDF5 behavior", location: "builtins::io::hdf5::tests::h5writeatt_h5info_and_h5disp_report_metadata" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

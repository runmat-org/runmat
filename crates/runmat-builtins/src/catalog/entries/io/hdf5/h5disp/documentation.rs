use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("h5disp"),
    slug: Some("h5disp"),
    summary: "Return a textual summary of an HDF5 file, group, or dataset.",
    description: "`h5disp(filename)` returns display text for the file root. `h5disp(filename, location)` returns display text for a selected group or dataset.",
    keywords: &["h5disp", "hdf5", "metadata", "display", "dataset", "group", "file", "io"],
    related: &["h5info"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The text traverses groups, datasets, shapes, datatypes, and attributes. It is returned as a string so scripts can capture or print it deterministically.", "HDF5 access uses the native HDF5 library. Browser execution reports `RunMat:hdf5:UnsupportedTarget`; there is no GPU kernel or fusion route. Admitted resident numeric controls are materialized on the host before native access."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Native HDF5 file access is synchronous host work. It terminates fusion and does not launch a GPU kernel. The active native filesystem path authority and HDF5 library must be available; a missing native route reports a typed error rather than falling back to an unrelated ambient path."] },
    ],
    examples: &[BuiltinExample {
        id: "native-h5disp",
        title: "h5disp with a deterministic native HDF5 file",
        program: "h5write('measurements.h5', '/group/A', [1 2; 3 4]);\ntxt = h5disp('measurements.h5');\ntxt",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(contains(txt, 'Group: /group')); assert(contains(txt, 'Dataset: /group/A'));" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can this run in a browser?", answer: "The current HDF5 backend requires a native HDF5 library; browser calls return RunMat:hdf5:UnsupportedTarget." },
    ],
    links: &[BuiltinDocumentationLink { label: "h5info", target: BuiltinDocumentationLinkTarget::Builtin("h5info") }],
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

use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("hdf5info"),
    slug: Some("hdf5info"),
    summary: "Legacy alias for HDF5 metadata traversal.",
    description: "`hdf5info(filename)` and `hdf5info(filename, location)` preserve the legacy spelling and return the same metadata structure as `h5info`.",
    keywords: &["hdf5info", "h5info", "hdf5", "metadata", "dataset", "group", "file", "io", "legacy"],
    related: &["h5info"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Root and group traversal is recursive. The root/group and dataset fields are the same as `h5info`.", "The alias uses the same native HDF5 host route, ownership, error identifiers, and browser unsupported-target behavior as `h5info`."] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &["Native HDF5 file access is synchronous host work. It terminates fusion and does not launch a GPU kernel. The active native filesystem path authority and HDF5 library must be available; a missing native route reports a typed error rather than falling back to an unrelated ambient path."] },
    ],
    examples: &[BuiltinExample {
        id: "native-hdf5info",
        title: "hdf5info with a deterministic native HDF5 file",
        program: "h5write('measurements.h5', '/group/A', [1 2]);\ninfo = hdf5info('measurements.h5');\ninfo.Groups",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(info.Name, '/')); assert(numel(info.Groups) == 1);" },
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
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Native HDF5 behavior", location: "builtins::io::hdf5::alias_tests::legacy_read_write_and_info_share_exact_native_dataset_semantics" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

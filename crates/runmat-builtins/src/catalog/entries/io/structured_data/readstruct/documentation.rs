use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("readstruct"), slug: Some("readstruct"),
    summary: "Read XML data into a scalar structure.",
    description: "`readstruct(filename)` reads XML from the active RunMat filesystem provider and projects its root into a scalar struct.",
    keywords: &["readstruct", "xml", "struct", "file"],
    related: &["xmlread", "xmlwrite"],
    sections: &[
        BuiltinDocumentationSection { heading: "XML projection", paragraphs: &[
            "Elements, text, attributes, and repeated siblings are supported. Attributes become fields named `<attribute><AttributeSuffix>`; text on non-leaf elements uses `TextNodeName`; repeated element names become cell-valued fields. Comments and the prolog are skipped, entities are unescaped, and malformed tags are rejected.",
            "Name-value controls are `FileType` (only `xml`), `AttributeSuffix`, and `TextNodeName`. HDF5, Parquet, JSON, schema-aware import, and full XML DOM behavior are not provided by this form.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The input path is resolved by the installed filesystem provider. Resident inputs are gathered to the host before parsing; the result is a host struct, not a GPU kernel or fusion result. The reader materializes the XML text and parsed tree in host memory.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "project-xml", title: "Read XML into a struct",
        program: "s = readstruct('config.xml');\nassert(strcmp(s.name, 'runmat'));",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "readstruct-xml" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "config.xml", content: BuiltinFixtureContent::Utf8("<config><name>runmat</name></config>") }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(s.name, 'runmat'));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What XML features are supported?", answer: "A lightweight XML projection with elements, attributes, text, repeated siblings, and malformed-tag checks; not schema-aware import." }],
    links: &[
        BuiltinDocumentationLink { label: "xmlread", target: BuiltinDocumentationLinkTarget::Builtin("xmlread") },
        BuiltinDocumentationLink { label: "xmlwrite", target: BuiltinDocumentationLinkTarget::Builtin("xmlwrite") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime entrypoint", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/xml.rs") },
            BuiltinDocumentationLink { label: "XML parser and projection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/xml/parser.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "XML struct projection", location: "builtins::io::repl_fs::xml::tests::readstruct_imports_xml_children_attributes_and_repeats" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

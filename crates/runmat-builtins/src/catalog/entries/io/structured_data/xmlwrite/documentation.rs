use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xmlwrite"), slug: Some("xmlwrite"),
    summary: "Serialize lightweight XML DOM-compatible values.",
    description: "`xmlwrite(document)` returns serialized XML as a character row; `xmlwrite(filename, document)` writes UTF-8 bytes through the installed filesystem provider and returns the byte count.",
    keywords: &["xmlwrite", "xml", "file", "dom"], related: &["xmlread", "readstruct"],
    sections: &[
        BuiltinDocumentationSection { heading: "Accepted documents", paragraphs: &[
            "Accepts the lightweight `org.w3c.dom.Document` and `org.w3c.dom.Element` objects returned by `xmlread`, scalar XML text (including a singleton string array), and scalar structs for simple file-emission workflows. Element serialization uses TagName, TextContent, Attributes, and ChildNodes. XML text and attribute values are escaped.",
            "This is not a full Java DOM implementation. It does not validate schemas or preserve original formatting after object mutation.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &[
            "The one-argument serialization form is host-only and does not access the filesystem. The two-argument form writes through the active filesystem provider. Resident arguments are materialized on the host before serialization. There is no GPU kernel, fusion route, or certified finite complete-call memory bound for the current serializer.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "write-document", title: "Write an XML document",
        program: "doc = xmlread('config.xml');\nn = xmlwrite('copy.xml', doc);\nassert(n > 0);",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "xmlwrite-document" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "config.xml", content: BuiltinFixtureContent::Utf8("<config><name>runmat</name></config>") }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "copy = xmlread('copy.xml'); assert(strcmp(copy.DocumentElement.TagName, 'config')); assert(n > 0);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does xmlwrite preserve original formatting?", answer: "No. It serializes the lightweight object tree and escapes text and attribute values; it does not preserve formatting or validate schemas." }],
    links: &[
        BuiltinDocumentationLink { label: "xmlread", target: BuiltinDocumentationLinkTarget::Builtin("xmlread") },
        BuiltinDocumentationLink { label: "readstruct", target: BuiltinDocumentationLinkTarget::Builtin("readstruct") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime entrypoint", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/xml.rs") },
            BuiltinDocumentationLink { label: "XML serializer", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/xml/serialize.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "XML serialization", location: "builtins::io::repl_fs::xml::tests::xmlwrite_serializes_dom_to_string_or_file" }],
        notes: &[],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

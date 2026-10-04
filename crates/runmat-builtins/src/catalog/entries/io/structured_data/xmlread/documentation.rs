use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xmlread"), slug: Some("xmlread"),
    summary: "Read XML text into a lightweight DOM-compatible object.",
    description: "Reads XML from the active RunMat filesystem provider into a lightweight document object with parsed element, attributes, child nodes, and text content.",
    keywords: &["xmlread", "xml", "file", "dom"], related: &["xmlwrite", "readstruct"],
    sections: &[
        BuiltinDocumentationSection { heading: "Document representation", paragraphs: &[
            "The `org.w3c.dom.Document`-compatible object has `Filename`, `Text`, `DocumentElementName`, and `DocumentElement` properties. Its `org.w3c.dom.Element`-compatible root has `NodeName`, `TagName`, `TextContent`, `Attributes`, and `ChildNodes`.",
            "Mismatched, unterminated, multiple, or missing root tags are rejected. This lightweight representation does not implement Java method dispatch or the full W3C DOM interface.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The input is gathered to the host when resident; the filesystem service supplies the file. Parsing materializes text and an object tree on the host. There is no GPU kernel or fusion route.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "document-root", title: "Inspect the document root",
        program: "doc = xmlread('config.xml');\nroot = doc.DocumentElement;\nassert(strcmp(root.TagName, 'config'));",
        display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "xmlread-document" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "config.xml", content: BuiltinFixtureContent::Utf8("<config><name>runmat</name></config>") }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(root.TagName, 'config')); assert(strcmp(doc.DocumentElementName, 'config'));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Is this a complete W3C DOM?", answer: "No. The object exposes a lightweight property-based compatibility surface, not Java method dispatch." }],
    links: &[
        BuiltinDocumentationLink { label: "xmlwrite", target: BuiltinDocumentationLinkTarget::Builtin("xmlwrite") },
        BuiltinDocumentationLink { label: "readstruct", target: BuiltinDocumentationLinkTarget::Builtin("readstruct") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime entrypoint", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/xml.rs") },
            BuiltinDocumentationLink { label: "XML parser", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/xml/parser.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Document object", location: "builtins::io::repl_fs::xml::tests::xmlread_returns_lightweight_document_object" }],
        notes: &[],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

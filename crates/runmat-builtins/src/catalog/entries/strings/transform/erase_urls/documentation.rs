use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Syntax and behavior", paragraphs: &[
        "newStr = eraseURLs(str) accepts a string array, character vector, or cell array of character vectors and returns the same documented container type. newDocuments = eraseURLs(documents) removes web-address tokens from tokenizedDocument input.",
        "The text form erases HTTP and HTTPS URLs. Scheme matching is case-insensitive. The tokenizedDocument form preserves document shape and remaining explicit token-type metadata in RunMat's current scalar compatibility-object representation.",
        "All integer and logical inputs are invalid; there is no numeric-to-text conversion. Multirow character matrices and cells containing strings or nested cells are separately gated RunMat-only extensions.",
        "The documented surface covers HTTP and HTTPS URLs; bare www addresses are not advertised as compatible behavior.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "eraseURLs is portable host text processing. It has no resident-text representation, provider kernel or fusion route. Invalid resident numeric values reject before provider access or gathering.",
        "The runtime binding handles text and tokenizedDocument without callbacks or external resources. The text transform uses a case-insensitive URL matcher; malformed input may throw.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-text", title: "Erase a URL from text", program: "str = \"For more information, see https://www.mathworks.com\";\nnewStr = eraseURLs(str)", display_output: Some("newStr = \"For more information, see \""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(newStr, \"For more information, see \"));" } },
    BuiltinExample { id: "legacy-documents", title: "Erase web-address tokens from documents", program: "documents = tokenizedDocument(\"Visit https://example.com now\");\nnewDocuments = eraseURLs(documents)", display_output: Some("The resulting document contains the tokens Visit and now."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(newDocuments.Documents{1}, [\"Visit\" \"now\"])); assert(isequal(newDocuments.NumDocuments, 1));" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/compat.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("eraseURLs"), slug: Some("eraseURLs"),
    summary: "Erase HTTP and HTTPS URLs from text and web-address tokens from tokenized documents.",
    description: "eraseURLs erases HTTP and HTTPS URLs from documented text containers. It also accepts tokenizedDocument input and removes tokens whose type is web-address.",
    keywords: &["eraseURLs", "URL", "HTTP", "HTTPS", "tokenizedDocument"], related: &["erasePunctuation", "tokenizedDocument"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does eraseURLs accept tokenizedDocument input?", answer: "Yes. It removes tokens whose type is web-address." }, BuiltinDocumentationFaq { question: "Does eraseURLs remove a bare www address?", answer: "The documented surface specifies HTTP and HTTPS URLs. RunMat does not advertise bare www matching as MATLAB-compatible behavior." }],
    links: &[BuiltinDocumentationLink { label: "MathWorks eraseURLs documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/textanalytics/ref/eraseurls.html") }, BuiltinDocumentationLink { label: "erasePunctuation", target: BuiltinDocumentationLinkTarget::Builtin("erasePunctuation") }, BuiltinDocumentationLink { label: "tokenizedDocument", target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument") }, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text admission and compatibility", location: "builtins::strings::transform::compat::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "document token removal", location: "builtins::strings::text_analytics::documents::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "identity inference", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/erase_urls/inference.rs" }], notes: &["The former syntax example and points are represented by the typed signatures and Syntax and behavior section. Both legacy examples now have executable result assertions. GPU, fusion, compatibility gates and resident-input behavior are represented in the catalog contract and placement section."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

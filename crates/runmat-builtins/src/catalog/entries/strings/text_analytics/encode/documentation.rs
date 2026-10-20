use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
        "counts = encode(bag, documents)\ncounts = encode(bag, words)\ncounts = encode(___, \"DocumentsIn\", \"rows\")\ncounts = encode(___, \"DocumentsIn\", \"columns\")\nC = encode(___, \"ForceCellOutput\", true)",
        "`counts` is a sparse count matrix. `C` is a 1-by-1 cell containing that sparse matrix.",
        "`bag` must be a scalar RunMat `bagOfWords` or `bagOfNgrams` object.",
        "`documents` must be a RunMat `tokenizedDocument` object.",
        "Non-tokenized input must be a row word vector and is treated as one document.",
        "`DocumentsIn` must be `'rows'` or `'columns'`.",
        "`ForceCellOutput` must be a logical scalar.",
    ] },
    BuiltinDocumentationSection { heading: "Inputs and orientation", paragraphs: &[
        "encode accepts a scalar RunMat bagOfWords or bagOfNgrams object and either tokenizedDocument input or a row word vector treated as one document. Integer and numeric model/document values are not converted to text and reject before provider access.",
        "The result is a host sparse double count matrix; documents are rows and bag terms are columns by default. DocumentsIn='columns' transposes that orientation. Unrecognized words or n-grams are ignored; bagOfNgrams uses the stored n-gram rows and their scan lengths.",
        "ForceCellOutput=true wraps the sparse result in a 1-by-1 cell. This scalar-bag form does not imply support for non-scalar bag object arrays. MATLAB-compatible mode requires a logical scalar; RunMat mode may accept exact scalar integer zero as false and nonzero as true through an extension gate.",
        "Non-scalar bag object arrays and native MATLAB sparse model storage remain outside the implemented compatibility surface.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "Bag models and documents are host object/text values. Resident numeric values in those roles reject before provider access; only a compatibility-gated resident ForceCellOutput control may gather after validation. There is no GPU encode kernel.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "bag-words", title: "Encode documents with a bag of words", program: "bag = bagOfWords(tokenizedDocument([\"alpha beta\"; \"beta gamma\"])); newDocuments = tokenizedDocument([\"beta beta delta\"; \"alpha gamma\"]); counts = encode(bag, newDocuments)", display_output: Some("A sparse 2-by-3 count matrix ordered by bag.Vocabulary."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(counts), [2 3]));" } },
    BuiltinExample { id: "documents-columns", title: "Put documents in columns", program: "bag = bagOfWords(tokenizedDocument(\"alpha beta\")); counts = encode(bag, [\"alpha\" \"beta\" \"alpha\"], \"DocumentsIn\", \"columns\")", display_output: Some("One column for the single word-vector document."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(size(counts, 2) == 1);" } },
    BuiltinExample { id: "ngrams", title: "Encode n-grams", program: "bag = bagOfNgrams(tokenizedDocument(\"a b a\"), \"NgramLengths\", [1 2]); counts = encode(bag, tokenizedDocument(\"a b a b\"))", display_output: Some("Sparse unigram and bigram counts for rows in bag.Ngrams."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(size(counts, 1) == 1);" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/encode.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("encode"), slug: Some("encode"), summary: "Encode documents against a bag model as sparse counts.", description: "encode creates a host sparse double count matrix from bagOfWords or bagOfNgrams and tokenized documents or row words.",
    keywords: &["encode", "bagOfWords", "bagOfNgrams", "text analytics", "sparse", "count matrix"], related: &["bagOfWords", "bagOfNgrams", "tokenizedDocument"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does encode return sparse output?", answer: "Yes. Both supported bag models produce a host sparse double matrix." },
        BuiltinDocumentationFaq { question: "Does ForceCellOutput implement bag object arrays?", answer: "No. It supports a scalar-bag 1-by-1 cell wrapper only." },
        BuiltinDocumentationFaq { question: "Does encode execute on the GPU?", answer: "No. Model and document roles are host object/text inputs; only a gated resident ForceCellOutput control may gather." },
    ], links: &[BuiltinDocumentationLink { label: "bagOfWords", target: BuiltinDocumentationLinkTarget::Builtin("bagOfWords") }, BuiltinDocumentationLink { label: "bagOfNgrams", target: BuiltinDocumentationLinkTarget::Builtin("bagOfNgrams") }, BuiltinDocumentationLink { label: "MathWorks encode", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/textanalytics/ref/bagofwords.encode.html") }, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "bag count, orientation and control tests", location: "builtins::strings::text_analytics::encode::tests" }], notes: &["Non-scalar bag arrays and MATLAB sparse model identity remain open."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

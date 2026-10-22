use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("cosineSimilarity"),

    slug: Some("cosineSimilarity"),

    summary: "Compute cosine similarity between documents, bag models, or numeric matrix rows.",

    description: "`cosineSimilarity` compares rows of numeric matrices directly, or compares `tokenizedDocument`, `bagOfWords`, and `bagOfNgrams` inputs using tf-idf weighting derived from the first document or bag input. MATLAB documents double numeric matrices; RunMat mode separately gates single, logical, typed-integer, and resident matrix extensions.",

    keywords: &["cosineSimilarity", "text analytics", "similarity", "tf-idf", "bagOfWords", "bagOfNgrams"],
    related: &["tokenizedDocument", "bagOfWords", "bagOfNgrams"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &["S = cosineSimilarity(documents)\nS = cosineSimilarity(documents, queries)\nS = cosineSimilarity(bag)\nS = cosineSimilarity(bag, queries)\nS = cosineSimilarity(M)\nS = cosineSimilarity(M1, M2)", "`S` is a sparse similarity matrix for real inputs, or a dense complex matrix for complex matrix inputs.", "Document and bag forms use tf-idf weights from the first input.", "Query documents are encoded against the first input's vocabulary or n-gram set.", "Matrix inputs compare rows and must have equal column counts."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`cosineSimilarity(M)` returns pairwise row cosine similarities for the rows of `M` as a real sparse matrix.",
                "`cosineSimilarity(M1, M2)` returns an `m1`-by-`m2` sparse matrix comparing each row of `M1` to each row of `M2`; both matrices must have the same number of columns.",
                "All eight real or typed-complex integer classes are a RunMat-only extension and require each component to be exactly representable at the binary64 norm and inner-product boundary; aligned values above `flintmax` are accepted when their significant bits fit binary64.",
                "Single, logical, and provider-resident numeric matrices are independent RunMat-only extensions.",
                "`cosineSimilarity(documents)` accepts RunMat `tokenizedDocument` objects and raw row word vectors; non-tokenized arrays must be row vectors representing one document.",
                "`cosineSimilarity(documents, queries)` encodes query documents using the vocabulary and inverse-document-frequency weights derived from `documents`.",
                "`cosineSimilarity(bag)` and `cosineSimilarity(bag, queries)` accept RunMat `bagOfWords` and `bagOfNgrams` compatibility objects and use their stored counts and terms.",
                "Real numeric and text-model results are returned as RunMat `SparseTensor` values. The public MATLAB reference does not specify zero-norm behavior; RunMat returns stored `NaN` similarities because the cosine quotient is mathematically undefined when either norm is zero.",
                "Complex matrix inputs use the Hermitian inner product. RunMat does not yet have a complex sparse value type, so complex results are returned as a dense `ComplexTensor` compatibility fallback rather than MATLAB's sparse complex matrix representation.",
                "`gpuArray` inputs are gathered before evaluation because RunMat sparse output is currently host-resident and provider-resident sparse similarity kernels are tracked by the GPU fast-path audit.",
                "Sparse-complex output, table adapters, and tall or distributed execution remain explicit general representation/container gaps rather than integer-storage gaps.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`cosineSimilarity` builds host sparse similarity matrices from text-model metadata or numeric rows. Provider-resident sparse output is not available in this slice.", "`cosineSimilarity` currently gathers provider inputs and returns host sparse output because RunMat sparse values are host-resident."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["Real results use host sparse double storage; complex results use dense complex double storage. This representation difference remains explicit. Text models use host vocabulary/count processing, not provider-resident sparse execution."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Compare Numeric Rows",
            program: "S = cosineSimilarity([1 0; 1 1])",
            display_output: Some("`S` is a sparse 2-by-2 matrix of row cosine similarities."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(issparse(S)); assert(isequal(size(S), [2 2])); assert(abs(S(1,1)-1) < 1e-12); assert(abs(S(2,2)-1) < 1e-12); assert(abs(S(1,2)-1/sqrt(2)) < 1e-12); assert(abs(S(2,1)-S(1,2)) < 1e-12);",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Compare Documents Against Queries",
            program: "documents = tokenizedDocument([\"alpha beta\"; \"beta gamma\"]);\nqueries = tokenizedDocument(\"alpha\");\nS = cosineSimilarity(documents, queries)",
            display_output: Some("`S` contains the similarity between each document and the query document."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(issparse(S)); assert(isequal(size(S), [2 1])); assert(S(1,1) > 0); assert(S(2,1) == 0);",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Compare A Bag Model",
            program: "bag = bagOfWords(tokenizedDocument([\"alpha beta\"; \"beta gamma\"]));\nS = cosineSimilarity(bag)",
            display_output: Some("`S` compares the bag rows using tf-idf weights derived from the bag counts."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(issparse(S)); assert(isequal(size(S), [2 2])); assert(abs(S(1,1)-1) < 1e-12); assert(abs(S(2,2)-1) < 1e-12); assert(abs(S(1,2)-S(2,1)) < 1e-12);",
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `cosineSimilarity` return a sparse matrix?",
            answer: "Yes for real numeric, document, and bag inputs. Complex numeric inputs currently return a dense complex matrix because RunMat does not yet expose complex sparse storage.",
        },
        BuiltinDocumentationFaq {
            question: "Can query documents contain words not in the first input?",
            answer: "Yes. Query words or n-grams outside the first input's vocabulary are ignored, matching the fixed-vocabulary comparison model.",
        },
        BuiltinDocumentationFaq {
            question: "Does `cosineSimilarity` execute on the GPU?",
            answer: "No. The current implementation returns host sparse values. Provider-resident sparse output and GPU cosine kernels belong to the queued GPU fast-path audit.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "MATLAB cosineSimilarity documentation",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/textanalytics/ref/cosinesimilarity.html"),
        },
        BuiltinDocumentationLink {
            label: "tokenizedDocument",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument"),
        },
        BuiltinDocumentationLink {
            label: "bagOfWords",
            target: BuiltinDocumentationLinkTarget::Builtin("bagOfWords"),
        },
        BuiltinDocumentationLink {
            label: "bagOfNgrams",
            target: BuiltinDocumentationLinkTarget::Builtin("bagOfNgrams"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/similarity.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/similarity.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::strings::text_analytics::similarity::tests",
        }],
        notes: &[],
    },
    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),
};

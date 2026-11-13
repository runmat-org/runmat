use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tsne"),
    slug: Some("tsne"),
    summary: "Embed high-dimensional observations using t-distributed stochastic neighbor embedding.",
    description: "`tsne(X)` embeds complete observation rows into a lower-dimensional numeric matrix. `[Y,loss] = tsne(...)` also returns the final Kullback-Leibler divergence.",
    keywords: &["tsne", "t-SNE", "dimensionality reduction", "embedding", "statistics", "machine learning"],
    related: &["svd", "kmeans", "pdist"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Observations and outputs",
            paragraphs: &[
                "`X` is a real single- or double-precision vector or 2-D matrix; the current RunMat runtime also accepts logical observations. Matrix rows are observations. A column vector has one predictor per row; a row vector is one observation with several predictors.",
                "Rows containing NaN are omitted before fitting, so the number of output rows can be smaller than the input row count. Inf is rejected. The embedding `Y` and scalar `loss` are double precision.",
                "`NumDimensions` selects the output width and defaults to 2. `InitialY` may contain all original input rows or only the complete rows that remain after NaN omission.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Optimization and distances",
            paragraphs: &[
                "`Algorithm` accepts `barneshut` and `exact`. The current implementation computes a bounded exact CPU embedding for both names; it does not implement Barnes-Hut acceleration.",
                "`Distance` accepts `euclidean`, `fasteuclidean`, `seuclidean`, `fastseuclidean`, `cityblock`, `chebychev`, `minkowski`, `mahalanobis`, `cosine`, `correlation`, `spearman`, `hamming`, and `jaccard`. Minkowski uses exponent 2. Function-handle distances are rejected.",
                "`Standardize` centers and scales predictors before distances. Positive `NumPCAComponents` below the predictor count applies PCA before pairwise distances.",
                "`Perplexity`, `Exaggeration`, `LearnRate`, `Theta`, `Verbose`, `NumPrint`, and `CacheSize` are validated. The exact CPU optimizer does not use Theta, Verbose, NumPrint, or CacheSize after validation. `Options` accepts statset-style MaxIter, TolFun, and Display; nonempty OutputFcn is rejected because callbacks are not invoked.",
                "The implementation rejects oversized dense exact workloads before allocation. Random initialization uses RunMat's RNG and is not bit-for-bit identical to MATLAB.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and residency",
            paragraphs: &[
                "Native integer observation data and integer numeric option values are RunMat extensions. Numerical integer values must cross a checked, exactly representable binary64 boundary; structural controls decode exactly.",
                "t-SNE executes on the host and has no device kernel or fusion path. Automatic residency may gather through its owning provider. Explicit gpuArray input is admitted only with RunMat language extensions and materializes a host result by contract.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Embed four observations",
            program: "X = [0 0; 0.2 0.1; 9.8 9.9; 10 10.1]; Y = tsne(X,'Perplexity',2)",
            display_output: Some("Y is a 4-by-2 embedding matrix."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Y),[4 2])); assert(strcmp(class(Y),'double'));" },
        },
        BuiltinExample {
            id: "example-2", title: "Request the final loss",
            program: "X = [0 0; 0.2 0.1; 9.8 9.9; 10 10.1]; [Y,loss] = tsne(X,'Algorithm','exact','Perplexity',2,'Options',struct('MaxIter',250))",
            display_output: Some("loss is the final Kullback-Leibler divergence."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Y),[4 2])); assert(isscalar(loss)); assert(isfinite(loss));" },
        },
        BuiltinExample {
            id: "example-3", title: "Omit incomplete observations",
            program: "Y = tsne([1 2; NaN 3; 10 11],'Perplexity',1)",
            display_output: Some("Y has two rows because the incomplete observation is omitted."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Y),[2 2]));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does RunMat implement Barnes-Hut acceleration?", answer: "No. The barneshut name is accepted, but the current bounded CPU solver uses the exact method for both algorithm settings." },
        BuiltinDocumentationFaq { question: "Are stochastic results bit-for-bit identical to MATLAB?", answer: "No. RunMat has its own RNG stream, so random initial embeddings are not bit-for-bit identical." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "svd", target: BuiltinDocumentationLinkTarget::Builtin("svd") },
        BuiltinDocumentationLink { label: "kmeans", target: BuiltinDocumentationLinkTarget::Builtin("kmeans") },
        BuiltinDocumentationLink { label: "pdist", target: BuiltinDocumentationLinkTarget::Builtin("pdist") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/tsne.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Embedding admission and materialization", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/tsne/admission.rs") },
            BuiltinDocumentationLink { label: "Exact embedding optimizer", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/tsne/optimizer.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Embedding and admission tests", location: "builtins::stats::ml::tsne::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "VM ML integration", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

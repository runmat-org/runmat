use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("kmeans"),
    slug: Some("kmeans"),
    summary: "Partition observations into k clusters with k-means clustering.",
    description: "`kmeans(X,k)` clusters observation rows and supports up to four outputs: `[idx,C,sumd,D] = kmeans(...)`.",
    keywords: &["kmeans", "k-means", "clustering", "statistics", "machine learning"],
    related: &["pdist", "pdist2", "fitctree"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Observations and outputs",
            paragraphs: &[
                "`X` must be a real numeric vector or 2-D matrix. Matrix rows are observations; row and column vectors are treated as one-dimensional observations. Rows containing NaN are omitted from fitting and receive NaN in the returned `idx` and `D`; Inf is rejected.",
                "Outputs are one-based cluster assignments `idx`, centroid matrix `C`, per-cluster distance sums `sumd`, and observation-by-cluster distances `D`.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Starts and optimization controls",
            paragraphs: &[
                "The default distance is squared Euclidean. Supported distances are `sqeuclidean`, `cityblock`, `cosine`, `correlation`, and `hamming`. Cityblock and Hamming use component-wise median centroids; other distances use mean centroids in the metric's transformed space.",
                "`Start` accepts `plus`, `sample`, `uniform`, `cluster`, a numeric k-by-p centroid matrix, or a k-by-p-by-r array of replicate starts. With numeric Start, `k` may be `[]`; it is then inferred from the first dimension.",
                "`Replicates` runs multiple starts and keeps the smallest finite within-cluster objective. Numeric 3-D starts set the replicate count from their third dimension. `MaxIter` limits Lloyd iterations; `EmptyAction` accepts `singleton`, `error`, or `drop`.",
                "`Display` is accepted for source compatibility. `OnlinePhase=on` runs an additional bounded CPU refinement phase. `Options` accepts statset-style structs: MaxIter and Display are honored; enabled parallel stream fields are rejected by the current single-process CPU implementation.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and residency",
            paragraphs: &[
                "Typed-integer observation data, cluster counts, numeric starts, iteration controls, and numeric parallel toggles are role-specific RunMat extensions. Structural controls decode exactly; numerical solver inputs must be exactly representable at the binary64 boundary.",
                "The clustering solver runs on CPU. Resident observations and consumed resident numeric controls are gathered through their owning providers. When `X` is resident, RunMat attempts to restore numeric outputs to the owner of `X`; automatic placement may fall back to host output if that provider cannot restore them. For resident `X`, explicit device intent on `X` or a consumed resident numeric control requires resident outputs or an error. With host `X`, outputs remain on the host. Ignored statset fields and empty statset metadata do not request device output residency. There is no `kmeans` device kernel or fusion path.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Cluster two groups",
            program: "X = [0 0; 0.2 0.1; 9.8 9.9; 10 10.1]; [idx,C,sumd,D] = kmeans(X,2,'Start',[0 0;10 10])",
            display_output: Some("The first two rows share one cluster, the last two share another, and C has two centroids."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(idx),[4 1])); assert(isequal(size(C),[2 2])); assert(isequal(size(sumd),[2 1])); assert(isequal(size(D),[4 2])); assert(idx(1)==idx(2)); assert(idx(3)==idx(4)); assert(idx(1)~=idx(3));" },
        },
        BuiltinExample {
            id: "example-2", title: "Use cityblock distance",
            program: "X = [0 0; 1 0; 9 10; 10 10]; idx = kmeans(X,2,'Distance','cityblock','Replicates',3)",
            display_output: Some("Three cityblock starts are compared; idx assigns each of the four observations."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(idx),[4 1])); assert(all(idx>=1 & idx<=2));" },
        },
        BuiltinExample {
            id: "example-3", title: "Infer k from numeric starts",
            program: "X = [0 0; 1 0; 9 10; 10 10]; [idx,C] = kmeans(X,[],'Start',[0 0;10 10])",
            display_output: Some("The two start rows establish two clusters."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(idx),[4 1])); assert(isequal(size(C),[2 2]));" },
        },
        BuiltinExample {
            id: "example-4", title: "Retain the position of a missing row",
            program: "idx = kmeans([1; 2; NaN; 20],2)",
            display_output: Some("The third assignment is NaN; complete rows are clustered."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(idx),[4 1])); assert(isnan(idx(3)));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does RunMat reproduce MATLAB's exact random starts?", answer: "No. RunMat uses its own RNG stream behind `rng`; stochastic starts are not bit-for-bit identical." },
        BuiltinDocumentationFaq { question: "Are custom distance functions supported?", answer: "No. Custom function-handle distances are rejected; the named distance set is documented above." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "pdist", target: BuiltinDocumentationLinkTarget::Builtin("pdist") },
        BuiltinDocumentationLink { label: "pdist2", target: BuiltinDocumentationLinkTarget::Builtin("pdist2") },
        BuiltinDocumentationLink { label: "fitctree", target: BuiltinDocumentationLinkTarget::Builtin("fitctree") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/kmeans.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Argument admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/kmeans/admission.rs") },
            BuiltinDocumentationLink { label: "Clustering solver", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/kmeans/compute.rs") },
            BuiltinDocumentationLink { label: "Output ownership and restoration", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/kmeans/output.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Clustering and admission tests", location: "builtins::stats::ml::kmeans::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "VM ML integration", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Regularized GLM implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lassoglm.rs"),
};

const SOLVER_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Regularized GLM solver",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lassoglm/solver.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("lassoglm"), slug: Some("lassoglm"),
    summary: "Fit lasso or elastic-net regularized generalized linear models.",
    description: "`lassoglm(X,Y,distr)` fits a regularized generalized linear model. Columns of `B` follow ascending Lambda values; a second output returns FitInfo.",
    keywords: &["lassoglm", "lasso", "elastic net", "generalized linear model", "glm", "binomial", "poisson", "statistics", "machine learning"],
    related: &["lasso", "mnrfit", "statset"],
    sections: &[
        BuiltinDocumentationSection { heading: "Data and distributions", paragraphs: &[
            "`X` is a nonempty finite real matrix with observations in rows. `distr` accepts normal (also spelled gaussian), binomial, and poisson. Normal and poisson responses are numeric vectors. Binomial responses may be a probability/logical vector or an N-by-2 successes/trials matrix. `BinomialSize` supplies scalar or per-observation trial counts for binomial probability responses.",
            "`Alpha` is in `(0,1]`; one gives lasso and smaller positive values mix in a ridge penalty. `Lambda` accepts nonnegative scalar or vector values, sorted and de-duplicated into ascending output columns. Without it, `NumLambda` and `LambdaRatio` define a geometric path from an all-zero-coefficient model. `Standardize`, `Intercept` (also `Constant`), nonnegative `Weights` with positive total weight, and finite observation `Offset` control the model.",
            "`Link` accepts only the canonical link for the selected distribution: identity, logit, or log. `EstDisp` currently accepts off only. Unsupported values reject rather than silently selecting another model. `RelTol`, `MaxIter`, and statset-style Options control convergence; Options.MaxIter, Options.TolX, and Options.TolFun are honored. Enabled parallel options reject because fitting is CPU-local.",
            "`CV` accepts resubstitution or a positive K-fold count. K-fold assignment currently uses deterministic round-robin partitions rather than MATLAB's random fold sequence. FitInfo then includes SE, LambdaMinDeviance, Lambda1SE, IndexMinDeviance, and Index1SE. FitInfo also contains Intercept, Lambda, Deviance, DF, and Iterations.",
        ] },
        BuiltinDocumentationSection { heading: "Classes and execution", paragraphs: &[
            "Typed-integer X/Y, floating parameters, direct controls, and honored nested Options fields are independently gated RunMat extensions. Integer values must be exactly representable where they enter the binary64 solver; lossy values reject. Coefficients and numeric FitInfo values are double.",
            "The regularized GLM solver runs on CPU. Admitted resident inputs are gathered through their owning provider; explicit gpuArray inputs require RunMat mode. Results are host values. No resident fitting kernel or fusion route is claimed. Callback-free fitting is portable to native and browser runtimes, subject to the same host memory and iteration limits.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "binomial", title: "Fit a binomial logistic model", program: "X = [0;1;2;3;4;5]; y = [0;0;0;1;1;1]; [B,FitInfo] = lassoglm(X,y,'binomial','Lambda',[0 0.1]); disp(B);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B),[1 2])); assert(numel(FitInfo.Intercept)==2); assert(isequal(FitInfo.Lambda,[0 0.1]));" } },
        BuiltinExample { id: "statset", title: "Control a Poisson fit with statset", program: "X = [0;1;2;3;4;5]; counts = [1;1;2;3;4;5]; opts = statset('lassoglm','MaxIter',400); [B,FitInfo] = lassoglm(X,counts,'poisson','Options',opts); disp(FitInfo.Lambda);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(size(B,1)==1); assert(numel(FitInfo.Lambda)==size(B,2));" } },
        BuiltinExample { id: "cross-validation", title: "Select a cross-validated penalty", program: "X = [0;1;2;3;4;5]; y = [1;3;5;7;9;11]; [B,FitInfo] = lassoglm(X,y,'normal','CV',3); coef = B(:,FitInfo.IndexMinDeviance); disp(coef);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(FitInfo.SE)==size(B,2)); assert(FitInfo.IndexMinDeviance>=1 && FitInfo.IndexMinDeviance<=size(B,2));" } },
        BuiltinExample { id: "grouped-binomial", title: "Fit grouped binomial observations", program: "X = [0;1;2]; Y = [8 10;3 10;1 8]; B = lassoglm(X,Y,'binomial','Weights',[1;2;1]); disp(B);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(size(B,1)==1); assert(size(B,2)>=1);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Where is the intercept?", answer: "B contains predictor coefficients only; FitInfo.Intercept has one intercept per retained Lambda." },
        BuiltinDocumentationFaq { question: "Which links are implemented?", answer: "Identity for normal, logit for binomial, and log for poisson. Other Link values reject." },
        BuiltinDocumentationFaq { question: "How are Lambda values ordered?", answer: "FitInfo.Lambda is ascending and B uses the same column order." },
        BuiltinDocumentationFaq { question: "Does K-fold assignment reproduce MATLAB's random partition?", answer: "No. The current solver assigns folds deterministically in round-robin order." },
    ],
    links: &[SOURCE, BuiltinDocumentationLink { label: "lasso", target: BuiltinDocumentationLinkTarget::Builtin("lasso") }, BuiltinDocumentationLink { label: "mnrfit", target: BuiltinDocumentationLinkTarget::Builtin("mnrfit") }, BuiltinDocumentationLink { label: "statset", target: BuiltinDocumentationLinkTarget::Builtin("statset") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE, SOLVER_SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "GLM options, solver and FitInfo", location: "crates/runmat-runtime/src/builtins/stats/ml/lassoglm/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM GLM forms", location: "crates/runmat-vm/tests/stats_ml.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM exact integer solver boundaries", location: "crates/runmat-vm/tests/integer_json_join_and_statistical_semantics.rs" },
    ], notes: &["Fitting uses a CPU regularized GLM solver with double coefficients and host FitInfo values."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

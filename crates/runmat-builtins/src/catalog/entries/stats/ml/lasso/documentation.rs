use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Lasso solver and argument admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lasso.rs"),
};

const SOLVER_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Coordinate-descent solver",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lasso/solver.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("lasso"),
    slug: Some("lasso"),
    summary: "Fit lasso or elastic-net regularized linear regression models.",
    description: "`lasso(X,y)` fits a regularized linear model. Each column of `B` corresponds to an ascending `Lambda` value; `[B,FitInfo]` also returns fit metadata.",
    keywords: &["lasso", "elastic net", "regularization", "regression", "statistics", "machine learning"],
    related: &["corr", "normalize", "lassoglm"],
    sections: &[
        BuiltinDocumentationSection { heading: "Data and model", paragraphs: &[
            "`X` is a nonempty finite real matrix with observations in rows; `y` is a finite real vector with one value per row. `Alpha` is in `(0,1]`: one gives lasso and smaller positive values mix in a ridge penalty. `Standardize` and `Intercept` control the model form. Disabling `Intercept` also disables standardization and yields zero intercepts.",
            "`Lambda` accepts nonnegative scalar or vector values; the solver sorts and de-duplicates them for ascending output columns. Without `Lambda`, `NumLambda` and `LambdaRatio` define a geometric path from the largest all-zero-model penalty. `DFmax` may remove fitted columns. `Weights` must be nonnegative with positive total weight and are normalized internally. `RelTol` and `MaxIter` control convergence; `FitInfo.Iterations` records work per retained Lambda.",
            "`CV` accepts resubstitution or a positive K-fold count. K-fold partitioning currently uses deterministic round-robin assignment, not MATLAB's random fold sequence. Cross-validation adds SE, LambdaMinMSE, Lambda1SE, IndexMinMSE, and Index1SE to FitInfo. `PredictorNames`, `DFmax`, `UseCovariance`, and `MCReps=1` are accepted; fitting uses direct coordinate descent, so FitInfo.UseCovariance is false. Tall-array ADMM and covariance-matrix acceleration are not implemented. CacheSize, AbsTol, B0, U0, Rho, and Options forms reject rather than being silently ignored.",
        ] },
        BuiltinDocumentationSection { heading: "Classes and execution", paragraphs: &[
            "Typed-integer X/y, floating parameters, structural controls, and logical controls have independent RunMat-only extension gates. Integer values cross a checked binary64 solver boundary; unrepresentable values reject rather than round. The coefficient and numeric FitInfo outputs are double.",
            "The coordinate-descent solver runs on CPU. Admitted resident inputs are gathered through their owning provider; explicit gpuArray inputs require RunMat mode. Results are host values. No resident fitting kernel or fusion is claimed. Callback-free fitting can execute in native or browser runtimes, subject to the same host memory and iteration limits.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "explicit-lambda", title: "Fit a sparse linear model", program: "X = [0 1;1 1;2 1;3 1;4 1]; y = [1;3;5;7;9]; B = lasso(X,y,'Lambda',0,'Standardize',false); disp(B);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B),[2 1])); assert(abs(B(1)-2)<1e-6); assert(abs(B(2))<1e-6);" } },
        BuiltinExample { id: "elastic-path", title: "Return an elastic-net fit path", program: "X = [0 1;1 1;2 1;3 1;4 1]; y = [1;3;5;7;9]; [B,FitInfo] = lasso(X,y,'Alpha',0.5,'Lambda',[0 0.01 0.1]); coef0 = FitInfo.Intercept(1); disp(coef0);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B),[2 3])); assert(isequal(FitInfo.Lambda,[0 0.01 0.1])); assert(numel(FitInfo.Intercept)==3);" } },
        BuiltinExample { id: "cross-validation", title: "Choose a cross-validated penalty", program: "X = [0 1;1 1;2 1;3 1;4 1;5 1]; y = [1;3;5;7;9;11]; [B,FitInfo] = lasso(X,y,'CV',3); idx = FitInfo.IndexMinMSE; coef = B(:,idx); disp(coef);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(idx>=1 && idx<=size(B,2)); assert(numel(FitInfo.SE)==size(B,2));" } },
        BuiltinExample { id: "weighted", title: "Weight observations", program: "X = [0 1;1 1;2 1;3 1;4 1]; y = [1;3;5;7;9]; w = ones(size(y)); w(1:5) = 2; [B,FitInfo] = lasso(X,y,'Weights',w); disp(FitInfo.Intercept);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(size(B,1)==2); assert(numel(FitInfo.Intercept)==size(B,2));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Where is the intercept?", answer: "B contains predictor coefficients only; FitInfo.Intercept has one intercept per retained Lambda." },
        BuiltinDocumentationFaq { question: "Does UseCovariance enable covariance-matrix acceleration?", answer: "No. The current solver uses direct coordinate descent and reports FitInfo.UseCovariance as false." },
        BuiltinDocumentationFaq { question: "How are Lambda values ordered?", answer: "FitInfo.Lambda is ascending and B uses the same column order." },
        BuiltinDocumentationFaq { question: "Does K-fold assignment reproduce MATLAB's random partition?", answer: "No. RunMat assigns folds deterministically in round-robin order; returned cross-validation fields and selection follow the documented solver behavior." },
    ],
    links: &[SOURCE, BuiltinDocumentationLink { label: "corr", target: BuiltinDocumentationLinkTarget::Builtin("corr") }, BuiltinDocumentationLink { label: "normalize", target: BuiltinDocumentationLinkTarget::Builtin("normalize") }, BuiltinDocumentationLink { label: "lassoglm", target: BuiltinDocumentationLinkTarget::Builtin("lassoglm") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE, SOLVER_SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Lasso options, solver and FitInfo", location: "crates/runmat-runtime/src/builtins/stats/ml/lasso/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM statistical semantics", location: "crates/runmat-vm/tests/integer_json_join_and_statistical_semantics.rs" },
    ], notes: &["Fitting uses a CPU coordinate-descent solver with double coefficients and host FitInfo values."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("lscov"),
    slug: Some("lscov"),
    summary: "Solve linear least-squares systems with observation covariance weighting.",
    description: "`lscov(A,B)` solves the linear least-squares system `A*x=B`. `lscov(A,B,V)` accepts observation weights or an observation covariance matrix. The optional algorithm is `\"chol\"` or `\"orth\"`.",
    keywords: &["lscov", "least squares", "weighted least squares", "generalized least squares", "statistics"],
    related: &["regress", "mldivide", "pinv"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Design, responses, and weights",
            paragraphs: &[
                "`A` is a nonempty two-dimensional design matrix. `B` is a response vector with one value per row of `A`, or a matrix with the same row count. Real and complex dense design and response values are admitted; sparse input and a provider-native solve are not supported. The coefficient output is size(A,2)-by-1 for a vector response and size(A,2)-by-size(B,2) for a matrix response. Host tensor conversion also admits typed integer values, but the floating-point solve may round wide integers; exact-integer behavior is not claimed for this form.",
                "`V` may be omitted or empty, a scalar weight, a nonnegative finite weight per observation (at least one positive), or a dense square observation covariance matrix. A covariance matrix must match size(A,1) and be symmetric positive semidefinite. The default `\"chol\"` form uses a Cholesky transform when positive definite and falls back to the SVD-backed orthogonal transform for singular covariance; `\"orth\"` selects that transform directly.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Outputs and execution",
            paragraphs: &[
                "The requested outputs are `[x,stdx,mse,S]`. `stdx` has the coefficient shape, `mse` is a 1-by-size(B,2) row, and the fourth output `S` is admitted only for a vector response. Complex coefficients and `S` retain complex values; standard errors and mean squared errors are real double.",
                "RunMat gathers admitted GPU-resident inputs to host before the dense weighted solve. There is no provider-native least-squares kernel or fusion route. Include a column of ones in `A` when the fit requires an intercept; `lscov` does not add one.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Fit an unweighted line",
            program: "A = [ones(3,1) (0:2)']; B = [1;3;5]; [x,stdx,mse,S] = lscov(A,B)",
            display_output: Some("x is approximately [1; 2] and mse is 0."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(x),[2 1])); assert(all(abs(x-[1;2])<1e-8)); assert(abs(mse)<1e-8); assert(isequal(size(stdx),[2 1])); assert(isequal(size(S),[2 2]));" },
        },
        BuiltinExample {
            id: "example-2", title: "Use observation weights",
            program: "A = [ones(3,1) (0:2)']; B = [1;2;10]; w = [1;1;100]; x = lscov(A,B,w)",
            display_output: Some("The high-weight third observation has more influence on x."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "u = lscov(A,B); assert(abs(A(3,:)*x-B(3))<abs(A(3,:)*u-B(3)));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does lscov add an intercept automatically?", answer: "No. Include a column of ones in A when the model should include an intercept." },
        BuiltinDocumentationFaq { question: "Does lscov run on the GPU?", answer: "No. Admitted GPU inputs are gathered and the dense weighted least-squares solve runs on the host." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "regress", target: BuiltinDocumentationLinkTarget::Builtin("regress") },
        BuiltinDocumentationLink { label: "mldivide", target: BuiltinDocumentationLinkTarget::Builtin("mldivide") },
        BuiltinDocumentationLink { label: "pinv", target: BuiltinDocumentationLinkTarget::Builtin("pinv") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lscov.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Input admission and weighting", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lscov/admission.rs") },
            BuiltinDocumentationLink { label: "Weighted fit", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lscov/fit.rs") },
            BuiltinDocumentationLink { label: "Least-squares solver", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lscov/solver.rs") },
            BuiltinDocumentationLink { label: "Covariance weighting transforms", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lscov/transform.rs") },
            BuiltinDocumentationLink { label: "Response geometry", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lscov/geometry.rs") },
            BuiltinDocumentationLink { label: "Resource checks", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/lscov/resources.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Weighted and complex least-squares semantics", location: "builtins::stats::ml::lscov::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

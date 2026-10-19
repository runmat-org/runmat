use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nlparci"),
    slug: Some("nlparci"),
    summary: "Confidence intervals for nonlinear least-squares parameter estimates.",
    description: "`nlparci(beta,r,'Jacobian',J)` or `nlparci(beta,r,'Covar',CovB)` returns lower and upper confidence bounds for each fitted parameter.",
    keywords: &["nlparci", "nonlinear regression", "confidence interval", "Jacobian", "covariance"],
    related: &["lsqnonlin", "lsqcurvefit", "tinv"],
    sections: &[
        BuiltinDocumentationSection { heading: "Input forms", paragraphs: &[
            "`beta` and `r` are real single or double vectors. `J` has one row per residual and one column per coefficient. `CovB` is a square estimated parameter covariance matrix with one row and column per coefficient. Optional `'Alpha',alpha` sets the significance level in the open interval (0,1); the default is 0.05.",
            "The covariance form is appropriate when the fitting procedure supplies its own covariance estimate, including robust fits. The Jacobian form estimates covariance from the complete residual sum of squares and the inverse normal equations using a rank-revealing SVD. NaN residuals, and rows with NaN Jacobian entries in the Jacobian form, are omitted as missing observations. Inf is rejected.",
        ] },
        BuiltinDocumentationSection { heading: "Intervals and execution", paragraphs: &[
            "The N-by-2 result contains lower bounds in column one and upper bounds in column two. Both forms use the two-sided Student-t critical value with complete-observation count minus parameter count degrees of freedom. At least one positive degree of freedom is required; the Jacobian must have full column rank. A zero residual variance gives a zero-width interval.",
            "The computation is CPU-owned and can suspend while gathering resident real floating inputs. Resident inputs are a RunMat-only extension and are rejected by MATLAB compatibility mode before provider access. It does not have a provider-native confidence-interval kernel or fuse with preceding GPU operations. Output class follows beta: a single beta produces single bounds; a double beta produces double bounds. Native integer, logical, complex, and sparse inputs are not accepted. Finite host-allocation admission for ordinary statistics calls remains an implementation limitation.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "jacobian-and-covariance", title: "Compute equivalent intervals from a Jacobian and covariance", program: "beta = [1; 2]; r = [1; -1; 1; -1]; J = [1 -1; 1 -1; 1 1; 1 1]; ciJ = nlparci(beta,r,'Jacobian',J); CovB = [0.5 0; 0 0.5]; ciC = nlparci(beta,r,'Covar',CovB);", display_output: Some("Both forms produce the same 2-by-2 intervals for this orthogonal Jacobian."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(ciJ),[2 2])); assert(all(abs(ciJ(:)-ciC(:))<1e-8)); assert(all(abs((ciJ(:,1)+ciJ(:,2))/2-beta)<1e-8));" } },
        BuiltinExample { id: "alpha", title: "Use a narrower confidence level", program: "beta = 1; r = [1; -1; 0]; J = [1; 1; 1]; ci95 = nlparci(beta,r,'Jacobian',J); ci90 = nlparci(beta,r,'Jacobian',J,'Alpha',0.1);", display_output: Some("The 90% interval is narrower than the 95% interval."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(ci90),[1 2])); assert(ci90(2)-ci90(1) < ci95(2)-ci95(1));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "When should I use Covar rather than Jacobian?", answer: "Use Covar when your fitting procedure supplies an estimated coefficient covariance matrix, particularly for robust fits. The Jacobian form estimates covariance from residuals and J." },
        BuiltinDocumentationFaq { question: "What happens when the Jacobian is rank deficient?", answer: "RunMat reports a numerical failure because finite ordinary confidence intervals are not identified by a rank-deficient Jacobian." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "MathWorks nlparci reference", target: BuiltinDocumentationLinkTarget::Source("https://www.mathworks.com/help/stats/nlparci.html") },
        BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/nlparci.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Interval calculation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/nlparci/compute.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Independent interval and invalid-input oracles", location: "builtins::stats::ml::nlparci::tests" }],
        notes: &["Official reference documents the two source forms, Alpha, missing NaN observations, output geometry, and full-rank validity condition. RunMat-specific rank-deficiency error and single-output-class policy require execution qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

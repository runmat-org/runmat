use crate::*;

const ENTRYPOINT_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Multinomial fitting entrypoint",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit.rs"),
};

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Multinomial solver implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/solver.rs"),
};
const ADMISSION_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Input and response admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/admission.rs"),
};
const DATA_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Predictor and design preparation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/data/prepare.rs"),
};
const RESPONSE_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Response class and grouped-count preparation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/data/response.rs"),
};
const TEXT_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Cooperative text class ordering and lookup",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/class_census/text.rs"),
};
const CATEGORICAL_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Borrowed categorical label decoding",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/containers/categorical/labels.rs"),
};
const PROBABILITY_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Nominal probabilities and likelihoods",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/solver/probability.rs"),
};
const LINEAR_ALGEBRA_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Information-system solve and covariance",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/solver/linear_algebra.rs"),
};
const OPTIONS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Typed option admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/options/runtime.rs"),
};
const OUTPUT_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Fit result construction",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/output.rs"),
};
const ALLOCATION_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Checked host workspace allocation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/mnrfit/allocation.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mnrfit"),
    slug: Some("mnrfit"),
    summary: "Fit nominal multinomial logistic regression coefficients.",
    description: "`mnrfit(X,Y)` fits nominal multinomial logistic regression coefficients. Rows of `X` are observations, columns are predictors, and columns of `B` correspond to all response classes except the final sorted reference class.",
    keywords: &["mnrfit", "multinomial", "logistic regression", "nominal", "binomial", "statistics", "machine learning"],
    related: &["lasso", "fitclinear"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and response classes", paragraphs: &[
            "`X` is a real numeric predictor matrix. Current MATLAB documents single or double `X`; typed-integer `X`, typed-integer `Y`, and logical `X` are independently gated RunMat extensions. Logical `Y` remains admitted. Integer predictors, grouped counts, weights, and tolerance values must be exactly representable at the binary64 host-solver boundary.",
            "`Y` may be a numeric, logical, string, character, cell-string, or categorical class vector with one response per row of `X`. Numeric classes retain distinct finite values exactly; classes are sorted and the last class is the reference. Alternatively, an N-by-K numeric `Y` supplies grouped response counts; columns are categories and row totals contribute observation weights. Rows with nonfinite `X` values or NaN response values are omitted. Infinite numeric class values are rejected; grouped count rows marked missing by NaN are omitted, while inspected non-NaN counts and their totals must be finite and nonnegative.",
        ] },
        BuiltinDocumentationSection { heading: "Solver controls and outputs", paragraphs: &[
            "The default model includes a constant term. `Model` supports `nominal`, `Link` supports `logit`, and `Interactions` accepts `off`. `Weights` is a nonnegative finite observation-weight vector with positive finite total after response weights are applied; unrepresentable products or totals return an error. `IterationLimit` and `Tolerance` control convergence. `EstDisp` accepts only `off`; overdispersion estimation is not implemented.",
            "`Constant`, `MaxIter`, `TolX`, `Options`, and `Display` remain gated RunMat option extensions rather than MATLAB-compatible spellings. `Display` is accepted but does not change the solver result.",
            "`B` has one row per model term and one column per non-reference class. `[B,dev,stats]` also returns deviance and a stats struct with `beta`, `se`, `t`, `p`, `covb`, `coeffcorr`, `dfe`, `s`, `estdisp`, and `classNames`. The solver runs on CPU; automatically resident inputs are gathered, while explicit gpuArray input is a mode-gated RunMat host-fallback extension. No resident solver kernel or fusion route is claimed.",
            "Long fits support execution cancellation. In the browser, preparation and solver work periodically yield to the event loop so cancellation requests can be processed.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "binary", title: "Fit a binary nominal logistic model",
            program: "X = [0; 1; 2; 3; 4]; y = [1; 2; 1; 2; 2]; [B,dev,stats] = mnrfit(X,y); disp(B); disp(dev);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B),[2 1])); assert(isequal(size(stats.se),size(B))); assert(isfinite(dev));" },
        },
        BuiltinExample { id: "three-class", title: "Fit a three-class nominal model",
            program: "X = [0; 1; 0; 1; 2; 3]; y = [1; 1; 2; 2; 3; 3]; B = mnrfit(X,y,'Model','nominal'); disp(B);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B),[2 2])); assert(all(isfinite(B(:))));" },
        },
        BuiltinExample { id: "grouped", title: "Fit grouped response counts",
            program: "X = [0; 1; 2; 3]; Y = [8 1 1; 5 3 2; 2 5 3; 1 6 4]; B = mnrfit(X,Y,'IterationLimit',200); disp(B);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B),[2 2])); assert(all(isfinite(B(:))));" },
        },
        BuiltinExample { id: "grouped-known-odds", title: "Check a grouped fit with known odds",
            program: "X = [0; 1; 2; 3]; Y = repmat([2 6],4,1); [B,dev] = mnrfit(X,Y,'Tolerance',1e-9); disp(B); disp(dev);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B),[2 1])); assert(strcmp(class(B),'double')); assert(abs(B(1)+log(3))<1e-6); assert(abs(B(2))<1e-6); assert(abs(dev)<1e-6);" },
        },
        BuiltinExample { id: "text", title: "Fit text response labels",
            program: "X = [0; 1; 2; 3; 4]; y = [\"low\"; \"high\"; \"low\"; \"high\"; \"high\"]; [B,dev,stats] = mnrfit(X,y,'Tolerance',1e-7); disp(stats.classNames);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isstring(stats.classNames)); assert(isequal(stats.classNames,[\"high\",\"low\"])); assert(isequal(size(B),[2 1])); assert(isfinite(dev));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which response class is the reference class?", answer: "RunMat sorts numeric, logical, and text response classes and uses the final sorted class as the reference for nominal coding. Grouped-count columns retain their supplied order, with the final column as the reference." },
        BuiltinDocumentationFaq { question: "Does RunMat implement ordinal models?", answer: "No. The current implementation fits nominal multinomial logistic regression; other Model values return an error." },
        BuiltinDocumentationFaq { question: "Does RunMat estimate overdispersion?", answer: "No. EstDisp accepts only off, and stats.s is 1." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "lasso", target: BuiltinDocumentationLinkTarget::Builtin("lasso") },
        BuiltinDocumentationLink { label: "fitclinear", target: BuiltinDocumentationLinkTarget::Builtin("fitclinear") },
        SOURCE,
        ENTRYPOINT_SOURCE,
        ADMISSION_SOURCE,
        DATA_SOURCE,
        RESPONSE_SOURCE,
        TEXT_SOURCE,
        CATEGORICAL_SOURCE,
        PROBABILITY_SOURCE,
        LINEAR_ALGEBRA_SOURCE,
        OPTIONS_SOURCE,
        OUTPUT_SOURCE,
        ALLOCATION_SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[ENTRYPOINT_SOURCE, SOURCE, ADMISSION_SOURCE, DATA_SOURCE, RESPONSE_SOURCE, TEXT_SOURCE, CATEGORICAL_SOURCE, PROBABILITY_SOURCE, LINEAR_ALGEBRA_SOURCE, OPTIONS_SOURCE, OUTPUT_SOURCE, ALLOCATION_SOURCE],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Multinomial fitting, class forms, and integer boundaries", location: "crates/runmat-runtime/src/builtins/stats/ml/mnrfit/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact numeric class identity, finite count totals, and weight boundaries", location: "crates/runmat-runtime/src/builtins/stats/ml/mnrfit/data/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Long-prefix text ordering, Unicode copies, and cancellation", location: "crates/runmat-runtime/src/builtins/stats/ml/mnrfit/class_census/text.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Borrowed categorical labels and undefined codes", location: "crates/runmat-runtime/src/builtins/table/containers/categorical/labels.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Browser task turns and public fit cancellation", location: "crates/runmat-wasm/src/runtime/execution/task_turn.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Provider failure and host admission", location: "crates/runmat-runtime/src/builtins/stats/ml/mnrfit/provider_tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Integer regression semantics", location: "crates/runmat-vm/tests/integer_normal_distribution_and_regression_semantics.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Cross-validation index implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/crossvalind.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("crossvalind"),
    slug: Some("crossvalind"),
    summary: "Generate cross-validation indices and logical train/test partitions.",
    description: "`crossvalind` provides MATLAB-compatible split helpers for legacy Statistics and Machine Learning workflows. K-fold calls return one-based numeric fold IDs; HoldOut, LeaveMOut, and Resubstitution return logical training/test masks.",
    keywords: &["crossvalind", "cross validation", "kfold", "holdout", "leavemout", "resubstitution", "statistics", "machine learning"],
    related: &["cvpartition", "training", "test"],
    sections: &[
        BuiltinDocumentationSection { heading: "Methods and outputs", paragraphs: &[
            "`crossvalind('KFold',n,k)` distributes `n` observations across `k` randomized folds and returns an n-by-1 double vector of IDs in `1:k`. KFold has one output. `crossvalind('HoldOut',n,p)` accepts a fraction in (0,1) or an integer test count in [1,n). `LeaveMOut` uses an integer count in [1,n); `Resubstitution` uses no partition value and makes both masks true. These three methods return an n-by-1 logical training mask by default, or `[train,test]` logical masks when two outputs are requested.",
            "`Classes` groups numeric, logical, string, string-array, or character-row labels with one label per observation; missing numeric labels remain ungrouped. `Min` requires enough observations in every class for the selected split. Labels retain native integer identity, including adjacent values above 2^53. The options are paired Name/Value arguments; Resubstitution does not accept Classes or Min.",
        ] },
        BuiltinDocumentationSection { heading: "Randomness and placement", paragraphs: &[
            "Random splits consume RunMat's shared runtime RNG. `rng(seed)` makes a given call reproducible. The helper gathers admitted resident inputs through their owners and creates host numeric or logical outputs; it has no resident split kernel or fusion path. Invalid host-visible method, arity, value, option, or requested-output controls are checked before random assignment.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "holdout", title: "Reproducible holdout mask", program: "rng(2026); idx = crossvalind('HoldOut', 10, 0.3); fprintf('%d\\n', numel(idx));", display_output: Some("10"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(idx)); assert(isequal(size(idx),[10 1])); assert(sum(~idx)==3);" } },
        BuiltinExample { id: "folds", title: "Tabulate randomized fold IDs", program: "rng(1); fold = crossvalind('KFold', 12, 3); tabulate(fold)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(fold),[12 1])); assert(all(fold>=1 & fold<=3)); assert(all([sum(fold==1),sum(fold==2),sum(fold==3)]==4));" } },
        BuiltinExample { id: "balanced-classes", title: "Balance classes in a holdout split", program: "rng(13); group = [1;1;1;1;2;2;2;2]; [train,test] = crossvalind('HoldOut', 8, 0.5, 'Classes', group); fprintf('%d %d\\n', sum(test(1:4)), sum(test(5:8)));", display_output: Some("2 2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(train) && islogical(test)); assert(all(train == ~test)); assert(sum(test(1:4))==2 && sum(test(5:8))==2);" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[SOURCE, BuiltinDocumentationLink { label: "cvpartition", target: BuiltinDocumentationLinkTarget::Builtin("cvpartition") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Cross-validation methods, classes, outputs, and exact labels", location: "crates/runmat-runtime/src/builtins/stats/ml/crossvalind.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM cross-validation forms", location: "crates/runmat-vm/tests/stats_ml.rs" }],
        notes: &["Native/browser, provider, and example-runner qualification remains required after integration."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

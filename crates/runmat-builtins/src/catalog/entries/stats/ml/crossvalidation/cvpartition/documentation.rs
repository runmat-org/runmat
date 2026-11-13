use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Cross-validation partition implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/cvpartition.rs"),
};

pub(super) const CONSTRUCTOR: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cvpartition"), slug: Some("cvpartition"),
    summary: "Create cross-validation partition objects.",
    description: "`cvpartition` creates an object for K-fold, holdout, leave-one-out, resubstitution, or custom test-set masks. Use `training` and `test` to retrieve logical masks.",
    keywords: &["cvpartition", "cross validation", "kfold", "holdout", "leaveout", "resubstitution", "statistics", "machine learning"],
    related: &["training", "test", "lasso"],
    sections: &[
        BuiltinDocumentationSection { heading: "Methods and properties", paragraphs: &[
            "`cvpartition(n,'KFold',k)` makes `k` balanced randomized folds for `n` observations, with `k` in `[2,n)`. `cvpartition(n,'Holdout',p)` accepts a fraction in `(0,1)` or an integer test count in `[1,n)`. `Leaveout` makes one test set per observation. `Resubstitution` creates one partition in which all observations appear in both returned training and test masks.",
            "`cvpartition('CustomPartition',testSets)` accepts a logical vector or matrix with one test set per column, or a positive-integer vector assigning each row a test-set number. The object exposes `NumObservations`, `NumTestSets`, `TestSize`, `TrainSize`, `Type`, `IsCustom`, `IsGrouped`, and `IsStratified`.",
        ] },
        BuiltinDocumentationSection { heading: "Stratification and compatibility", paragraphs: &[
            "`cvpartition(stratvar,'KFold',k)` and `cvpartition(stratvar,'Holdout',p)` stratify by default. Supply `'Stratify',false` to partition by observation order. Missing numeric labels remain counted in `NumObservations`, but their rows are false in training and test masks. Numeric, logical, string-array, and character-row labels are supported; typed integer labels retain exact native identity.",
            "Typed-integer observation counts, partition controls, stratification labels, and custom test sets are independently gated RunMat extensions. The documented `Stratify` value is a logical scalar in MATLAB compatibility mode; broad numeric, text, or integer boolean aliases are RunMat-only. Support is not yet claimed for `GroupingVariables`, tall-array holdout, GPU stratification/custom partitions, or all object summary and repartition forms.",
        ] },
        BuiltinDocumentationSection { heading: "Randomness and placement", paragraphs: &[
            "K-fold and holdout splits consume RunMat's shared runtime RNG; `rng(seed)` reproduces their masks. Resident constructor inputs are gathered through their owning provider. The split object and its metadata are host-side; no resident partition kernel or fusion is claimed.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "kfold-masks", title: "Create K-fold masks", program: "rng(7); c = cvpartition(6, 'KFold', 3); idxTest = test(c, 1); idxTrainAll = training(c, 'all'); fprintf('%d %d\\n', numel(idxTest), size(idxTrainAll,2));", display_output: Some("6 3"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(idxTest) && isequal(size(idxTest),[6 1])); assert(islogical(idxTrainAll) && isequal(size(idxTrainAll),[6 3]));" } },
        BuiltinExample { id: "stratified-holdout", title: "Stratified holdout", program: "rng(11); group = [1;1;2;2]; c = cvpartition(group, 'Holdout', 0.5); idx = test(c); fprintf('%d\\n', sum(idx));", display_output: Some("2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(idx),[4 1])); assert(sum(idx(1:2))==1 && sum(idx(3:4))==1);" } },
        BuiltinExample { id: "custom-test-sets", title: "Use custom test-set columns", program: "sets = logical([1 0; 0 1; 0 0]); c = cvpartition('CustomPartition', sets); idx = test(c, 'all'); disp(idx);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(idx,sets)); assert(c.NumTestSets==2);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does RunMat randomize partitions?", answer: "Yes. K-fold and holdout assignment consumes the shared RNG; set a seed to reproduce masks." },
        BuiltinDocumentationFaq { question: "How do I get masks for all folds?", answer: "Use `test(c,'all')` or `training(c,'all')`; columns correspond to test sets." },
    ],
    links: &[SOURCE, BuiltinDocumentationLink { label: "training", target: BuiltinDocumentationLinkTarget::Builtin("training") }, BuiltinDocumentationLink { label: "test", target: BuiltinDocumentationLinkTarget::Builtin("test") }, BuiltinDocumentationLink { label: "lasso", target: BuiltinDocumentationLinkTarget::Builtin("lasso") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Partition methods, options, and mask output", location: "crates/runmat-runtime/src/builtins/stats/ml/cvpartition.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM cross-validation forms", location: "crates/runmat-vm/tests/stats_ml.rs" }], notes: &["Provider, native/browser, and executable-example qualification remains required after integration."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const TEST: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("test"), slug: Some("test"), summary: "Return test-set masks from a cvpartition object.",
    description: "`test(c)` returns a logical mask for a single-test-set partition. `test(c,i)` selects one or more one-based K-fold, leaveout, or custom test-set columns; `test(c,'all')` returns every column.",
    keywords: &["test", "cvpartition", "cross validation", "mask", "statistics", "machine learning"],
    related: &["cvpartition", "training"],
    sections: &[BuiltinDocumentationSection { heading: "Indices and residency", paragraphs: &[
        "The index accepts documented single- or double-precision scalar or vector forms. RunMat mode additionally accepts logical and all eight typed-integer classes; integer indices are decoded exactly without conversion through double. A resident index is gathered through its owning provider and the logical output is restored to that provider. No resident mask-computation kernel or fusion is claimed.",
        "`test(c)` works for holdout, resubstitution, and one-column custom partitions. For a resubstitution partition, every row is true in both `test(c)` and `training(c)`.",
    ] }],
    examples: &[BuiltinExample { id: "one-fold", title: "Get one test fold", program: "rng(7); c = cvpartition(6, 'KFold', 3); idx = test(c, 2); fprintf('%d\\n', sum(idx));", display_output: Some("2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(idx)); assert(isequal(size(idx),[6 1]));" } }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How do I retrieve all test masks?", answer: "Use `test(c,'all')`; columns correspond to test-set indices." },
        BuiltinDocumentationFaq { question: "Can `i` be an integer array?", answer: "In RunMat mode, yes; every integer class is read exactly. MATLAB-compatible mode accepts the documented single- and double-precision forms." },
        BuiltinDocumentationFaq { question: "Does `test` preserve GPU residency?", answer: "When `i` is resident, RunMat gathers it through its owning provider and restores the logical output there." },
    ],
    links: &[SOURCE, BuiltinDocumentationLink { label: "cvpartition", target: BuiltinDocumentationLinkTarget::Builtin("cvpartition") }, BuiltinDocumentationLink { label: "training", target: BuiltinDocumentationLinkTarget::Builtin("training") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Mask index and resident-output tests", location: "crates/runmat-runtime/src/builtins/stats/ml/cvpartition.rs" }], notes: &["Provider qualification and example execution remain pending integration."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const TRAINING: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("training"), slug: Some("training"), summary: "Return training-set masks from a cvpartition object.",
    description: "`training(c)` returns a logical mask for a single-test-set partition. `training(c,i)` selects one or more one-based K-fold, leaveout, or custom test-set columns; `training(c,'all')` returns every column.",
    keywords: &["training", "cvpartition", "cross validation", "mask", "statistics", "machine learning"],
    related: &["cvpartition", "test"],
    sections: &[BuiltinDocumentationSection { heading: "Selection", paragraphs: &[
        "For each selected test set, training observations are normally the active-row complement of its test mask. Resubstitution is different: every observation is included in both returned masks. K-fold partitions have several test sets, so `training(c)` requires an index or `'all'`.",
        "Numeric single/double indices are documented forms. RunMat mode additionally accepts logical and typed-integer indices with exact structural decoding. A resident index is gathered through its owning provider and the logical result is restored to that provider.",
    ] }],
    examples: &[BuiltinExample { id: "all-folds", title: "Get every training fold", program: "rng(7); c = cvpartition(6, 'KFold', 3); idx = training(c, 'all'); fprintf('%d %d\\n', size(idx,1), size(idx,2));", display_output: Some("6 3"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(idx)); assert(isequal(size(idx),[6 3])); assert(all(sum(idx,1)==4));" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Why does `training(c)` error for K-fold partitions?", answer: "K-fold partitions have multiple test sets; specify a fold or use `'all'`." }],
    links: &[SOURCE, BuiltinDocumentationLink { label: "cvpartition", target: BuiltinDocumentationLinkTarget::Builtin("cvpartition") }, BuiltinDocumentationLink { label: "test", target: BuiltinDocumentationLinkTarget::Builtin("test") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Training mask tests", location: "crates/runmat-runtime/src/builtins/stats/ml/cvpartition.rs" }], notes: &["Provider qualification and example execution remain pending integration."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

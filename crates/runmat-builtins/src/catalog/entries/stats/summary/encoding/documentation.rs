use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Encoding runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/encoding.rs"),
};
const LABELS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Encoding labels and exact class identity",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/encoding/labels.rs"),
};
const CONTROLS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Encoding option admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/encoding/controls.rs"),
};
const GEOMETRY_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Encoding array geometry",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/encoding/geometry.rs"),
};
const OUTPUT_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Encoding typed output publication",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/encoding/output.rs"),
};

pub(super) const DUMMYVAR: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dummyvar"), slug: Some("dummyvar"),
    summary: "Create dummy-variable columns for grouping variables.",
    description: "`dummyvar(group)` converts grouping variables into a double design matrix with one column per level.",
    keywords: &["dummyvar", "dummy variables", "one hot", "categorical", "statistics"],
    related: &["onehotencode", "onehotdecode"],
    sections: &[
        BuiltinDocumentationSection { heading: "Grouping variables", paragraphs: &[
            "Numeric vectors use positive integer levels `1:max(group)`. Numeric row and column vectors are one grouping variable; a nonscalar numeric matrix contributes one grouping variable per column. The resulting dummy columns are concatenated.",
            "Numeric `NaN` rows produce `NaN` across that variable's dummy columns. Empty text and typed missing strings are also missing labels; literal `<missing>` text is an ordinary label. Cell-string vectors are ordered by first appearance; categorical inputs use the full category order, including unobserved categories.",
            "Cell arrays can supply several grouping variables as a row cell array. String, character-row, logical, and cell-string grouping variables are supported by RunMat.",
        ] },
        BuiltinDocumentationSection { heading: "Typed integers and GPU input", paragraphs: &[
            "Single and double numeric groups are documented. All eight typed-integer grouping classes are a compatibility-gated RunMat extension. Integer levels are checked in exact storage against the feasible output-size bound before conversion; infeasible wide levels error rather than rounding. Output remains double.",
            "Resident GPU grouping input is a separate RunMat extension. When enabled, the owning provider gathers it before CPU encoding and the result is a host double matrix. With extensions disabled, the call is rejected before provider access. There is no resident encoding kernel or fusion path.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "numeric-groups", title: "Encode numeric group levels",
        program: "D = dummyvar([1; 2; 1]); disp(D);",
        display_output: Some("[1 0; 0 1; 1 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(D, [1 0; 0 1; 1 0])); assert(strcmp(class(D), 'double'));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does `dummyvar` support missing labels?", answer: "Yes. Numeric `NaN`, empty text, and typed missing strings are missing labels; missing numeric rows produce `NaN` dummy values." }],
    links: &[
        BuiltinDocumentationLink { label: "onehotencode", target: BuiltinDocumentationLinkTarget::Builtin("onehotencode") },
        BuiltinDocumentationLink { label: "onehotdecode", target: BuiltinDocumentationLinkTarget::Builtin("onehotdecode") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE, LABELS_SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Numeric, missing, typed-integer and compatibility admission tests", location: "crates/runmat-runtime/src/builtins/stats/summary/encoding/tests/dummyvar.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled encoding surface", location: "crates/runmat-vm/tests/stats_ml.rs" },
    ], notes: &["Host-only double output; GPU input is a compatibility-gated gather boundary."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ONEHOTENCODE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("onehotencode"), slug: Some("onehotencode"),
    summary: "Encode class labels as one-hot vectors.",
    description: "`onehotencode(A,featureDim)` expands labels in `A` into one-hot vectors along `featureDim`.",
    keywords: &["onehotencode", "one hot", "classification", "categorical", "statistics"],
    related: &["onehotdecode", "dummyvar"],
    sections: &[
        BuiltinDocumentationSection { heading: "Dimensions and classes", paragraphs: &[
            "`featureDim` must identify a singleton dimension in `A`; dimensions beyond `ndims(A)` are padded with singleton dimensions. Categorical input infers classes from its full category order, including unobserved classes. Noncategorical input requires `ClassNames`, whose values must be unique and match the input label type.",
            "Numeric labels and `ClassNames` retain exact native identity. Adjacent `int64` and `uint64` labels above `flintmax` remain distinct.",
        ] },
        BuiltinDocumentationSection { heading: "Output class and missing labels", paragraphs: &[
            "The positional typename selects `double`, `single`, `logical`, or any of the eight integer output classes. `OutputType` and `Typename` name-value aliases are RunMat extensions.",
            "Missing labels and labels outside `ClassNames` encode as all-`NaN` feature vectors with floating output. Logical and integer output cannot represent those vectors and returns an error.",
            "Class discovery and matching run on the CPU. Automatic residency gathers through the owning provider; explicit `gpuArray` host fallback is a declared RunMat extension. Encoding is not fused or GPU-resident.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "string-classes", title: "Encode string labels along columns",
        program: "B = onehotencode([\"red\";\"blue\";\"red\"], 2, 'ClassNames', [\"blue\";\"red\"], 'OutputType', 'logical'); disp(B);",
        display_output: Some("[0 1; 1 0; 0 1]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, logical([0 1; 1 0; 0 1]))); assert(strcmp(class(B), 'logical'));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can I control the class order?", answer: "Yes. Supply `ClassNames` for noncategorical input, or use the category order of categorical input." }],
    links: &[
        BuiltinDocumentationLink { label: "onehotdecode", target: BuiltinDocumentationLinkTarget::Builtin("onehotdecode") },
        BuiltinDocumentationLink { label: "dummyvar", target: BuiltinDocumentationLinkTarget::Builtin("dummyvar") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE, CONTROLS_SOURCE, LABELS_SOURCE, GEOMETRY_SOURCE, OUTPUT_SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Class order, output type, exact integer labels, missing and extension tests", location: "crates/runmat-runtime/src/builtins/stats/summary/encoding/tests/encode.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled wide integer label semantics", location: "crates/runmat-vm/tests/integer_label_and_text_control_semantics.rs" },
    ], notes: &["Automatic GPU input is gathered; no provider-resident encoder is claimed."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ONEHOTDECODE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("onehotdecode"), slug: Some("onehotdecode"),
    summary: "Decode one-hot vectors into class labels.",
    description: "`onehotdecode(B,classes,featureDim)` selects the first maximum along the feature dimension and maps its index to `classes`.",
    keywords: &["onehotdecode", "one hot", "classification", "categorical", "statistics"],
    related: &["onehotencode", "dummyvar"],
    sections: &[
        BuiltinDocumentationSection { heading: "Selection and output shape", paragraphs: &[
            "`classes` must have as many entries as `B` has along `featureDim`. Ties, including an all-zero feature vector, select the first class. Output keeps the encoded shape with the feature dimension set to one.",
            "The default output is categorical and preserves the supplied class order as category order. A positional typename can request `double`, `single`, all eight integer classes, `string`, or `categorical`. Logical and cell-string outputs, plus `OutputType` and `Typename` aliases, are RunMat extensions.",
            "Native integer class labels retain exact identity during selection and requested native output conversion, including adjacent 64-bit values above `flintmax`.",
        ] },
        BuiltinDocumentationSection { heading: "Input and placement", paragraphs: &[
            "Documented encoded values are single or double in `[0,1]`. Checked integer or logical zero-one encoded arrays are a RunMat extension. Decoding runs on the CPU; automatic residency gathers transparently, and explicit `gpuArray` host fallback is separately compatibility gated.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "string-output", title: "Decode logical one-hot rows",
        program: "A = onehotdecode(logical([0 1; 1 0]), [\"blue\" \"red\"], 2, 'string'); disp(A);",
        display_output: Some("[\"red\"; \"blue\"]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(A, [\"red\"; \"blue\"]));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What happens when a vector contains all zeros?", answer: "The first class is selected because all classes tie at zero." }],
    links: &[
        BuiltinDocumentationLink { label: "onehotencode", target: BuiltinDocumentationLinkTarget::Builtin("onehotencode") },
        BuiltinDocumentationLink { label: "dummyvar", target: BuiltinDocumentationLinkTarget::Builtin("dummyvar") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE, CONTROLS_SOURCE, LABELS_SOURCE, GEOMETRY_SOURCE, OUTPUT_SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "First-max, exact integer classes, categorical order and compatibility tests", location: "crates/runmat-runtime/src/builtins/stats/summary/encoding/tests/decode.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled wide integer label semantics", location: "crates/runmat-vm/tests/integer_label_and_text_control_semantics.rs" },
    ], notes: &["Decoding is a host materialization boundary, not a GPU kernel or fusion path."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

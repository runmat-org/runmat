use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime dlarray construction",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/training.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dlarray"),
    slug: Some("dlarray"),
    summary: "Wrap floating-point or logical data in a labeled deep-learning array.",
    description: "`dlarray(data, format)` constructs a host `dlarray` object whose `Data` property retains the input's class, shape, and residency. `Format` and `Labels` store optional dimension-label text. Native integer Data and non-array objects are rejected.",
    keywords: &["dlarray", "deep learning", "array labels", "gpuArray", "autodiff"],
    related: &["dlfeval", "dlgradient", "dlnetwork", "forward"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Data and labels",
            paragraphs: &[
                "Host `double`, `single`, complex, and logical values are accepted. A valid provider-resident gpuArray remains in its original provider-owned handle inside `Data`; construction does not gather it. Native integer classes are not silently widened. The optional second argument must be host text, and this implementation stores its text as both `Format` and `Labels`.",
                "Additional constructor options are rejected. Supported arithmetic retains the dlarray wrapper and its dimension-label metadata.",
                "The object is a value-backed wrapper. When a `dlfeval` autodiff tape is active, supported host values are recorded as tape nodes. That is an ambient runtime-state effect, not pure construction. Provider-resident Data constructs without a download outside the tape, but an active tape rejects it with `RunMat:deepLearning:Unsupported` until resident tape kernels exist.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Shape and placement",
            paragraphs: &[
                "`size`, `numel`, rank, and shape predicates on a `dlarray` read the geometry of its `Data` property. For resident Data, these structural queries use handle metadata without downloading numerical contents. This guarantee is for the standard `dlarray` declaration, not for arbitrary project-defined external classes.",
                "Text, cells, structs, and foreign objects cannot be Data. Invalid Data or Format rejects before provider work, and valid Data is not cloned into a second numeric buffer.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "labeled-shape",
        title: "Wrap and inspect an array",
        program: "X = dlarray(ones(3,4), 'CB'); fprintf('%d %d\\n', size(X,1), size(X,2));",
        display_output: Some("3 4"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(class(X), 'dlarray')); assert(isequal(size(X), [3 4])); assert(strcmp(X.Format, 'CB'));",
        },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does dlarray download gpuArray Data?",
            answer: "Outside an active autodiff tape, construction retains the provider handle without downloading data; structural shape queries read its metadata. An active tape rejects resident Data with Unsupported until resident tape kernels are available.",
        },
        BuiltinDocumentationFaq {
            question: "Can Data contain typed integers or text?",
            answer: "Not in this implementation. Numeric floating-point, complex, and logical storage is accepted; native integer and non-array storage rejects without conversion.",
        },
    ],
    links: &[SOURCE, BuiltinDocumentationLink { label: "dlfeval", target: BuiltinDocumentationLinkTarget::Builtin("dlfeval") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "dlarray admission and resident ownership",
            location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs",
        }],
        notes: &["The documented standard-class shape projection is distinct from arbitrary external-class shape declarations. Integrated native and browser tests remain the release gate."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "RunMat cross-entropy implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/losses/mod.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("crossentropy"),
    slug: Some("crossentropy"),
    summary: "Compute cross-entropy loss for numeric predictions and targets.",
    description: "`crossentropy(Y,targets)` computes cross-entropy over finite, nonempty, equally sized host numeric arrays, host `dlarray` objects, or plain real resident `gpuArray` tensors. The default is a scalar loss; `Reduction='none'` preserves the prediction shape.",
    keywords: &["crossentropy", "deep learning", "loss"],
    related: &["dlarray", "trainnet"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Loss, weights, and formats",
            paragraphs: &[
                "Predictions are clipped inside the open unit interval before logarithms. Targets must be probabilities in [0,1]. An optional third positional argument or `Weights` name-value option supplies scalar, full prediction-sized, channel-sized, or observation-sized weights. A full-size logical or binary numeric `Mask` excludes selected terms.",
                "Options are `ClassificationMode`, `TargetCategories`, `Reduction`, `NormalizationFactor`, `DataFormat`, `WeightsFormat`, `Weights`, and `Mask`. The mode accepts single-label/exclusive and multi-label/independent aliases. Reduction accepts `sum` and `none`; `none` preserves the prediction shape. `DataFormat` and `WeightsFormat` use the supported `S`, `C`, `B`, `T`, and `U` labels; `WeightsFormat` maps weight axes by label. Normalization accepts `all-elements`, `batch-size`, `mask-included`, `none`, or a positive scalar. The default is batch-size when a format is present and all-elements otherwise.",
                "A host `dlarray` prediction produces an unformatted host `dlarray` scalar or array. Plain host numeric predictions produce a numeric result. A pair of plain numeric prediction and target arrays is a RunMat extension: MATLAB's `dlarray` crossentropy reference requires a `dlarray` target when predictions are numeric. Native integer host arrays are also a RunMat extension and are converted for floating-point loss arithmetic; wide integer values can round.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Resident execution",
            paragraphs: &[
                "Matching plain real resident prediction and target tensors can run through the provider cross-entropy kernel. Supported host scalar, full-size, channel, and observation weights or masks are uploaded in the prediction's single or double precision as temporary provider inputs; full-size resident weights and masks remain resident. With `Reduction='none'`, supported provider execution returns resident losses, preserving explicit device intent from any resident input. Scalar reduction runs on the provider and reads back only the scalar. Unsupported forms with automatically placed inputs may gather to the host after admission. An explicit `gpuArray` input instead returns an unsupported-operation error before a download; it is never silently gathered for host execution.",
                "GPU-backed `dlarray`, datastore/category target expansion, automatic differentiation, GPU `mask-included` normalization, and resident weight remapping with `WeightsFormat` are not implemented in this path. Provider support and the selected options determine whether a given call stays resident.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "scalar-loss",
            title: "Compute loss",
            program: "loss = crossentropy([0.8 0.2], [1 0]); fprintf('%d\\n', isscalar(loss) && loss > 0 && loss < 1);",
            display_output: Some("1"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isscalar(loss)); assert(abs(loss + log(0.8) / 2) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "class-weights",
            title: "Apply class weights",
            program: "Y = [0.8; 0.2]; T = [1; 0]; weights = [2; 1]; loss = crossentropy(Y, T, weights, 'DataFormat', 'CB'); fprintf('%d\\n', isscalar(loss) && loss > 0);",
            display_output: Some("1"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isscalar(loss)); assert(isfinite(loss));",
            },
        },
        BuiltinExample {
            id: "unreduced-multilabel",
            title: "Inspect per-element losses",
            program: "Y = [0.8 0.2]; T = [1 0]; losses = crossentropy(Y, T, 'ClassificationMode', 'multi-label', 'Reduction', 'none'); fprintf('%d\\n', numel(losses));",
            display_output: Some("2"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(losses), size(Y))); assert(all(isfinite(losses(:))));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        SOURCE,
        BuiltinDocumentationLink {
            label: "MATLAB crossentropy reference",
            target: BuiltinDocumentationLinkTarget::External(
                "https://www.mathworks.com/help/deeplearning/ref/dlarray.crossentropy.html",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Host, weighted, and provider-resident loss forms",
            location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

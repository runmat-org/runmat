use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime Adam update",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/training/adam/mod.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("adamupdate"),
    slug: Some("adamupdate"),
    summary: "Apply an Adam update to parameters and moving-average state.",
    description: "`adamupdate(params, grad, averageGrad, averageSqGrad, iteration, ...)` applies one Adam optimizer step to matching numeric arrays. Up to three results return updated parameters, first moment, and second moment.",
    keywords: &["adamupdate", "deep learning", "Adam", "optimizer", "moving average"],
    related: &["dlarray", "dlgradient", "dlupdate", "trainingOptions"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Inputs and outputs",
            paragraphs: &[
                "Parameters, gradients, and nonempty moment arrays must have matching shapes. Empty moment arrays initialize to zero using the parameter representation. Host numeric tensors must be double or single; host dlarray wrappers are supported around admitted host Data. A scalar numeric value is admitted, but integer tensors are not converted into floating-point optimizer state.",
                "`iteration` is a positive integer decoded exactly. Optional positional controls are `learnRate`, `gradDecay`, `sqGradDecay`, and `epsilon`, defaulting to 0.001, 0.9, 0.999, and 1e-8. Learning rate and epsilon must be positive; each decay factor is in [0, 1). MATLAB documents single and double controls; RunMat mode also accepts typed integer scalars, while MATLAB compatibility mode rejects that extension. The arithmetic is floating point, so a very large exactly decoded iteration is then used in a floating-point bias correction.",
                "An unbracketed call returns updated parameters. Bracketed calls may request zero to three outputs; a discarded result still validates and executes the update. A request for more than three outputs fails before provider work.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Provider-backed arrays",
            paragraphs: &[
                "Matching plain real gpuArray parameters, gradients, and resident moment arrays use the acceleration provider's Adam hook. Empty host moment arrays are initialized on device; successful provider execution returns provider-owned gpuArray outputs without downloading the full arrays.",
                "When a provider does not support the admitted Adam form, automatically resident inputs can fall back to a host update by gathering the operands. An explicit `gpuArray` input instead rejects without downloading, preserving its device intent. Successful resident results inherit explicit intent when any consumed resident input is explicit. Host dlarray is supported, but GPU-backed dlarray wrappers, model parameter structures, and training-loop callbacks are not supported by this function.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "one-scalar-step",
        title: "Update a scalar parameter",
        program: "[p, m, v] = adamupdate(2, 1, 0, 0, 1); fprintf('%.6f %.6f %.6f\\n', p, m, v);",
        display_output: Some("1.999000 0.100000 0.001000"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(p - 1.999) < 1e-9); assert(abs(m - 0.1) < 1e-12); assert(abs(v - 0.001) < 1e-12);",
        },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What does an empty moving-average input mean?",
            answer: "It initializes that moment to zeros with the parameter shape and representation before computing the step.",
        },
        BuiltinDocumentationFaq {
            question: "Does an update of gpuArray inputs always stay on the GPU?",
            answer: "Automatically resident inputs may gather for a supported host fallback. An explicitly created gpuArray requires a supported resident provider path and otherwise rejects without gathering; successful resident outputs retain explicit input intent.",
        },
    ],
    links: &[
        SOURCE,
        BuiltinDocumentationLink {
            label: "MATLAB adamupdate reference",
            target: BuiltinDocumentationLinkTarget::External(
                "https://www.mathworks.com/help/deeplearning/ref/adamupdate.html",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Host precision and provider-owned Adam updates",
            location: "crates/runmat-runtime/src/builtins/deep_learning/training/adam/tests.rs",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

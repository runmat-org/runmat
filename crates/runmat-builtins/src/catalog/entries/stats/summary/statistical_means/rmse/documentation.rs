use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rmse"), slug: Some("rmse"),
    summary: "Compute root mean squared error between forecast and actual arrays.",
    description: "`rmse(F,A)` reduces squared magnitudes of the broadcast-compatible difference. A dimension, dimension vector, or `\"all\"` selects reduction axes. `\"Weights\",W` adds nonnegative weights for a single operating dimension.",
    keywords: &["rmse", "root mean squared error", "forecast", "statistics"],
    related: &["rms", "mean"],
    sections: &[
        BuiltinDocumentationSection { heading: "Numerical contract", paragraphs: &[
            "Real and complex single/double forecast and actual inputs are supported. The output is real. Typed real or complex integer data and typed integer weights are separately mode-gated RunMat extensions.",
            "Weighted RMSE computes `sqrt(sum(W.*abs(F-A).^2)/sum(W))` in each selected slice. Weights may be a scalar, a full-size array, or a vector along the operating dimension. Zero total weight and all-omitted slices produce NaN. Weights cannot accompany `\"all\"` or a dimension vector.",
            "Missing flags `\"includenan\"`/`\"includemissing\"` and `\"omitnan\"`/`\"omitmissing\"` have paired behavior.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The host implementation retains scaled real or complex residual components until each slice is reduced. Eligible unweighted, same-owner floating GPU pairs enter a resident route; unsupported provider forms preserve the documented host-materialization behavior for GPU inputs. Resident weighted and mixed-input routes and performance qualification remain open.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "basic-error", title: "Root mean squared error",
            program: "predicted = [2 4 6]; actual = [1 4 8]; e = rmse(predicted, actual)",
            display_output: Some("e = sqrt(5/3)"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(e-sqrt(5/3)) < 1e-12);" },
        },
        BuiltinExample {
            id: "weighted-columns", title: "Weighted column-wise error",
            program: "F = [2 4; 4 8]; A = [1 4; 2 6]; W = [1; 3]; e = rmse(F, A, 1, \"Weights\", W)",
            display_output: Some("e = [sqrt(13/4) sqrt(3)]"), compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(e-[sqrt(13/4) sqrt(3)])) < 1e-12);" },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "How are weights normalized?",
        answer: "Weighted RMSE uses `sqrt(sum(W.*E.^2)/sum(W))` over each reduction slice.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "rms", target: BuiltinDocumentationLinkTarget::Builtin("rms") },
        BuiltinDocumentationLink { label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/statistical_means/rmse.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/statistical_means/rmse.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Statistical reduction classes, values and residency", location: "crates/runmat-runtime/src/builtins/stats/summary/statistical_means/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

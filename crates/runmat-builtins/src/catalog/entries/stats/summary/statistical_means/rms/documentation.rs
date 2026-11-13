use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rms"), slug: Some("rms"),
    summary: "Compute the root mean square magnitude of an array.",
    description: "`rms(X)` computes the root mean square magnitude along the first non-singleton dimension. Use `\"all\"`, a dimension, or a dimension vector for another reduction. `\"includenan\"`/`\"includemissing\"` include NaNs (the default); `\"omitnan\"`/`\"omitmissing\"` exclude them from each slice.",
    keywords: &["rms", "root mean square", "reduction", "signal"],
    related: &["rmse", "mean"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes and shapes", paragraphs: &[
            "Single and double, real or complex, are supported. Logical and character inputs enter double computation. Real and complex fixed-width integer inputs are a RunMat-mode extension and return double.",
            "The result is real. Selected dimensions become singleton dimensions. Empty slices and slices with no observations after omission return NaN.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The host implementation uses scaled component accumulation to avoid intermediate overflow. Eligible floating GPU arrays use a per-slice scaled resident route; unsupported provider forms preserve the documented host-materialization behavior for GPU inputs. Resident performance and edge-case qualification remain in progress.",
        ]},
    ],
    examples: &[BuiltinExample {
        id: "vector-rms", title: "Root mean square of a vector",
        program: "r = rms([1 2 4], \"all\")",
        display_output: Some("r = sqrt(7)"), compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(abs(r-sqrt(7)) < 1e-12);" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Is rms the same as norm divided by sqrt(n)?",
        answer: "For a vector without omitted values, yes: `rms(x)` equals `norm(x)/sqrt(numel(x))`.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "rmse", target: BuiltinDocumentationLinkTarget::Builtin("rmse") },
        BuiltinDocumentationLink { label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/statistical_means/rms.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/statistical_means/rms.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Statistical reduction classes, values and residency", location: "crates/runmat-runtime/src/builtins/stats/summary/statistical_means/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

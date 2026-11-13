use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("prctile"), slug: Some("prctile"),
    summary: "Compute sample percentiles.",
    description: "`prctile(X,p)` computes sample percentiles along the first nonsingleton dimension. The result is double precision. The requested percentages replace the operated dimension in the output shape.",
    keywords: &["prctile", "percentile", "quantile", "statistics"],
    related: &["quantile", "median"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`p` contains percentages in the closed interval [0,100]. A scalar request gives one result per slice; a vector request replaces the operated dimension with its number of elements.",
            "`prctile(X,p,dim)` selects a dimension, `prctile(X,p,vecdim)` reduces several dimensions, and `prctile(X,p,\"all\")` treats the entire input as one sample. The default is the first nonsingleton dimension.",
            "NaNs are omitted by default. `\"omitnan\"` keeps this behavior; `\"includenan\"` produces NaN for a slice containing NaN.",
            "`\"Method\",\"midpoint\"` is the default; `\"exact\"` remains the pre-R2025a alias. `\"inclusive\"` and `\"exclusive\"` correspond to the methods introduced in R2025a. RunMat also retains its older `\"linear\"` alias for inclusive. Approximate percentiles require unsupported tall arrays.",
        ]},
        BuiltinDocumentationSection { heading: "Classes and execution", paragraphs: &[
            "Real numeric data and requests are accepted. Single-precision input currently produces a double result. Typed-integer data and typed-integer percentage requests are separately gated RunMat extensions; integer data is ordered exactly before interpolation.",
            "Resident arrays are gathered through the owning provider. The operation is a materializing fusion boundary; it does not run a percentile GPU kernel.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "median-percentile", title: "Median percentile", program: "m = prctile([1 3 10 20], 50)",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(m-6.5)<1e-12);" },
        },
        BuiltinExample {
            id: "percentile-dimension", title: "Percentiles down columns", program: "p = prctile([1 10; 3 20], [0 50 100], 1)",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p), [3 2])); assert(max(abs(p(:)-[1;2;3;10;15;20]))<1e-12);" },
        },
        BuiltinExample {
            id: "percentile-vecdim", title: "Percentiles across multiple dimensions", program: "x = reshape(1:8, [2 2 2]); p = prctile(x, 50, [1 2])",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(p), [1 1 2])); assert(isequal(p(:), [2.5;6.5]));" },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Can p be a vector?", answer: "Yes. A vector request replaces the operated dimension with the number of requested percentiles." }],
    links: &[
        BuiltinDocumentationLink { label: "quantile", target: BuiltinDocumentationLinkTarget::Builtin("quantile") },
        BuiltinDocumentationLink { label: "median", target: BuiltinDocumentationLinkTarget::Builtin("median") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/order_stats.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Percentile admission and interpolation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/order_stats/quantiles.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Percentile shape, interpolation and integer compatibility", location: "crates/runmat-runtime/src/builtins/stats/summary/order_stats/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

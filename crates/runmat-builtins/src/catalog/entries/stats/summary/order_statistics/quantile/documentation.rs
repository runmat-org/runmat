use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("quantile"), slug: Some("quantile"),
    summary: "Compute sample quantiles.",
    description: "`quantile(X,p)` computes sample quantiles along the first nonsingleton dimension. The result is double precision. The requested probabilities replace the operated dimension in the output shape.",
    keywords: &["quantile", "percentile", "statistics", "order"],
    related: &["prctile", "median"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`p` contains probabilities in the closed interval [0,1]. A scalar integer `n` greater than one requests `n` evenly spaced quantiles; a vector of probabilities replaces the operated dimension with its number of elements.",
            "`quantile(X,p,dim)` selects a dimension, `quantile(X,p,vecdim)` reduces multiple dimensions, and `quantile(X,p,\"all\")` treats the entire input as one sample. The default is the first nonsingleton dimension.",
            "NaNs are omitted by default. `\"omitnan\"` keeps this behavior; `\"includenan\"` produces NaN for a slice containing NaN.",
            "`\"Method\",\"midpoint\"` is the default; `\"exact\"` remains the pre-R2025a alias. `\"inclusive\"` and `\"exclusive\"` correspond to the methods introduced in R2025a. RunMat also retains its older `\"linear\"` alias for inclusive. Approximate quantiles require unsupported tall arrays.",
        ]},
        BuiltinDocumentationSection { heading: "Classes and execution", paragraphs: &[
            "Real numeric data and requests are accepted. Single-precision input currently produces a double result. Typed-integer data and typed-integer probability requests are separately gated RunMat extensions; integer data is ordered exactly before interpolation.",
            "Resident arrays are gathered through the owning provider. The operation is a materializing fusion boundary; it does not run a quantile GPU kernel.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "quartiles", title: "Quartiles", program: "q = quantile([1 2 4 8], [0.25 0.5 0.75])",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(q), [1 3])); assert(max(abs(q(:)-[1.5;3;6]))<1e-12);" },
        },
        BuiltinExample {
            id: "nan-options", title: "Include or omit NaNs", program: "x = [1 NaN 3]; q = quantile(x, 0.5, \"includenan\"); r = quantile(x, 0.5)",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isnan(q)); assert(r==2);" },
        },
        BuiltinExample {
            id: "count-request", title: "Request an evenly spaced quantile count", program: "q = quantile([1 2 3 4 5], 4)",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(q), [1 4])); assert(max(abs(q-[1.5 2.5 3.5 4.5]))<1e-12);" },
        },
        BuiltinExample {
            id: "multi-dimension", title: "Reduce over a vector of dimensions", program: "x = reshape(1:8, [2 2 2]); q = quantile(x, 0.5, [1 2])",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(q), [1 1 2])); assert(isequal(q(:), [2.5;6.5]));" },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How is prctile different?", answer: "For supported forms, `prctile(X,p)` uses percentages and corresponds to `quantile(X,p/100)`." }],
    links: &[
        BuiltinDocumentationLink { label: "prctile", target: BuiltinDocumentationLinkTarget::Builtin("prctile") },
        BuiltinDocumentationLink { label: "median", target: BuiltinDocumentationLinkTarget::Builtin("median") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/order_stats.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Quantile admission and interpolation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/order_stats/quantiles.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Quantile interpolation, options and integer compatibility", location: "crates/runmat-runtime/src/builtins/stats/summary/order_stats/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

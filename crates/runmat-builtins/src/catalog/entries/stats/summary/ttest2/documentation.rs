//! User-facing two-sample test contract and examples.
use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/hypothesis.rs",
    ),
};

const ENGINE_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Two-sample statistics and output geometry",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/hypothesis/ttest2_engine.rs"),
};
const OPTIONS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Typed control admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/hypothesis/ttest2_options.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ttest2"),
    slug: Some("ttest2"),
    summary: "Perform a two-sample Student's t-test.",
    description: "`ttest2(x,y)` tests independent samples for equal means. The default assumes equal variances; `\"Vartype\",\"unequal\"` selects Welch's test.",
    keywords: &["ttest2", "t-test", "hypothesis test", "welch", "statistics", "student t"],
    related: &["tcdf", "tinv", "tpdf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Samples and dimensions", paragraphs: &[
            "Single- and double-precision vectors, matrices, and N-D arrays are supported. Vectors are single samples and may have different lengths. Arrays are tested along their first nonsingleton dimension unless `\"Dim\"` selects another; all other dimensions must agree.",
            "Dim is a positive, platform-representable integer. Dimensions beyond the input rank are singleton dimensions, subject to representable metadata geometry and successful allocation; there is no fixed 1024-dimension limit.",
            "NaNs are omitted separately from the two samples. If either sample has fewer than two valid observations, `h` is false and the numeric outputs are NaN.",
            "Native integer samples and integer `Alpha` or `Dim` controls are independently gated RunMat extensions. Statistical conversion to binary64 rejects values that cannot be represented exactly.",
        ] },
        BuiltinDocumentationSection { heading: "Options and outputs", paragraphs: &[
            "`Alpha` is a scalar in (0,1). `Tail` accepts `\"both\"`, `\"right\"`, or `\"left\"`; `\"larger\"` and `\"smaller\"` are accepted aliases. `Vartype` accepts `\"equal\"` and `\"unequal\"`.",
            "The four outputs are a logical test decision `h`, double p-value `p`, double confidence interval `ci`, and a struct with double `tstat`, `df`, and `sd` fields.",
        ] },
        BuiltinDocumentationSection { heading: "Resident inputs", paragraphs: &[
            "GPU samples are gathered through their owning provider for host reductions and Student-t distribution calculations. Outputs are host values; this operation does not use a resident statistical kernel.",
            "Sample and option transfers retain the original explicit execution context and typed cancellation. Existing integer compatibility gates and owning-provider resident-input forms are unchanged. Finite host-memory execution refuses before transfers or workspaces because the materialization, lane and result storage have no complete allocation certificate.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "equal-variance", title: "Default two-sample test",
            program: "[h,p,ci,stats] = ttest2([1;2;3;4], [2;4;6;8]); disp(h); disp(p);",
            display_output: Some("h and p for an equal-variance two-sample test"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(h)); assert(isscalar(p)); assert(numel(ci) == 2); assert(isfield(stats, 'tstat'));" },
        },
        BuiltinExample { id: "welch-left", title: "Welch one-sided test",
            program: "x = [1;2;3;4]; y = [2;4;6;8]; [h,p] = ttest2(x, y, \"Vartype\", \"unequal\", \"Tail\", \"left\"); disp(p);",
            display_output: Some("A left-tailed Welch p-value"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(h)); assert(p >= 0 && p <= 1);" },
        },
        BuiltinExample { id: "rows", title: "Test rows independently",
            program: "A = [1 2 3; 2 3 4]; B = [2 4 6; 3 5 7]; h = ttest2(A, B, \"Dim\", 2); disp(h);",
            display_output: Some("One logical decision per row"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(h)); assert(isequal(size(h), [2 1]));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does ttest2 implement Welch's test?", answer: "Yes. Pass `\"Vartype\",\"unequal\"` to use Satterthwaite degrees of freedom." },
        BuiltinDocumentationFaq { question: "What if a sample has fewer than two valid observations?", answer: "The corresponding test returns false and NaN numeric outputs." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tcdf", target: BuiltinDocumentationLinkTarget::Builtin("tcdf") },
        BuiltinDocumentationLink { label: "tinv", target: BuiltinDocumentationLinkTarget::Builtin("tinv") },
        BuiltinDocumentationLink { label: "tpdf", target: BuiltinDocumentationLinkTarget::Builtin("tpdf") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, ENGINE_SOURCE, OPTIONS_SOURCE],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
                label: "Equal and unequal variance, tail, missing, dimension, and integer tests",
                location: "crates/runmat-runtime/src/builtins/stats/summary/hypothesis.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Two-sample test executes from a MATLAB-syntax script",
                location: "crates/runmat-vm/tests/stats_ml.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
                label: "Provider admission, gather causes and borrowed-input lifetime",
                location: "crates/runmat-runtime/src/builtins/stats/summary/hypothesis/ttest2_provider_tests.rs" },
        ],
        notes: &["Computation and result publication occur on the host after owning-provider gather."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

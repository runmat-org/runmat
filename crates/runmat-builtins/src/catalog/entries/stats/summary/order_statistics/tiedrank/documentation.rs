use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tiedrank"), slug: Some("tiedrank"),
    summary: "Rank observations using average ranks for ties.",
    description: "`tiedrank(X)` assigns ascending average ranks to real numeric observations and preserves the input shape. `[R,tieadj] = tiedrank(X)` returns tie corrections; optional flags select Kendall corrections or symmetric Ansari scores.",
    keywords: &["tiedrank", "rank", "ties", "statistics"],
    related: &["corr", "sort"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Equal values receive the average of their occupied ranks. NaN input positions remain NaN in the output.",
            "Vector inputs are ranked across all elements. Matrix inputs are ranked within each column. N-dimensional inputs are ranked along the second dimension.",
            "For each tie group of size t, `tieadj` accumulates (t^3-t)/2. It is scalar for vectors and has the ranked dimension reduced to length one for arrays.",
            "`tiedrank(X,1)` requests Kendall corrections; its second output has three components, including pair and triplet corrections. `tiedrank(X,0,1)` assigns Ansari's symmetric positional scores, averaging occupied scores across ties.",
            "A call requesting no outputs performs input validation and returns no values; additional requested outputs currently receive scalar-zero padding.",
        ]},
        BuiltinDocumentationSection { heading: "Classes and execution", paragraphs: &[
            "Ranks and tie adjustments are double values. All eight real typed-integer data classes are RunMat extensions; exact same-class comparisons preserve ties above flintmax.",
            "Resident data is gathered through its owning provider. Ranking is a materializing fusion boundary, not a direct GPU kernel.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "average-ties", title: "Average tied ranks", program: "r = tiedrank([10 20 20 NaN])",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(r), [1 4])); assert(isequal(r(1:3), [1 2.5 2.5])); assert(isnan(r(4)));" },
        },
        BuiltinExample {
            id: "tie-adjustment", title: "Tie adjustment", program: "[r, t] = tiedrank([3; 1; 1])",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(r, [3;1.5;1.5])); assert(t==3);" },
        },
        BuiltinExample {
            id: "kendall-correction", title: "Kendall tie corrections", program: "[r, t] = tiedrank([-2 1 3 1 4], 1)",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(r, [1 2.5 4 2.5 5])); assert(isequal(t(:), [3;0;18]));" },
        },
        BuiltinExample {
            id: "ansari-scores", title: "Ansari symmetric ranks", program: "r = tiedrank([-2 1 3 1 4], 0, 1)",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(r, [1 2.5 2 2.5 1]));" },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How are ties handled?", answer: "All observations in a tie group receive the average occupied rank. The second output sums (t^3-t)/2 for tie groups in each ranked slice." }],
    links: &[
        BuiltinDocumentationLink { label: "corr", target: BuiltinDocumentationLinkTarget::Builtin("corr") },
        BuiltinDocumentationLink { label: "sort", target: BuiltinDocumentationLinkTarget::Builtin("sort") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/order_stats.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Ranking and tie corrections", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/order_stats/ranks.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Rank, tie adjustment and wide-integer semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/order_stats/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

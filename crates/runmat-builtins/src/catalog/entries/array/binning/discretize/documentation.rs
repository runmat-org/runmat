use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Numeric bins and included edges",
        paragraphs: &[
            "`discretize(X,edges)` returns the one-based bin containing each element of `X`. By default, bins include their left edge and exclude their right edge, except that the final bin includes both edges. `IncludedEdge=\"right\"` reverses that convention while keeping the first outer edge included.",
            "Edges must be monotonically increasing and may contain consecutive repeated values. Values outside the outer edges and `NaN` values receive `NaN` when the output contains default bin indices.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Computed edges and replacement values",
        paragraphs: &[
            "A positive scalar `N` asks RunMat to compute `N` uniform-width double bins. The two-output form returns the computed edge row as `E`; a second output is not available when explicit edges are supplied.",
            "A numeric or text vector after the edges replaces bin indices. Numeric output preserves the replacement vector's class. Out-of-range and missing inputs become `NaN` for floating replacement values, exact zero for fixed-width integer replacement values, and an empty string for text replacement values.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current capability boundary",
        paragraphs: &[
            "This implementation currently accepts real numeric and logical `X`, numeric or logical edges, numeric or text replacement values, and the `IncludedEdge` option. Datetime, duration, calendar-duration bin widths, and categorical output are not implemented yet.",
            "Provider-resident inputs are gathered and the result is host-resident. This differs from the compatible GPU-array contract, which retains supported work on the GPU; the catalog records the current host-placement boundary rather than claiming device parity.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "explicit-edges",
        title: "Assign values to explicit bins",
        program: "Y = discretize([0.2 1.5 2.0], [0 1 2])",
        display_output: Some("Y = [1 2 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Y, [1 2 2]));" },
    },
    BuiltinExample {
        id: "right-edge",
        title: "Include the right edge",
        program: "Y = discretize([1 3 4 7 10 11], [1 3 4 7 10 11], \"IncludedEdge\", \"right\")",
        display_output: Some("Y = [1 1 2 3 4 5]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Y, [1 1 2 3 4 5]));" },
    },
    BuiltinExample {
        id: "computed-edges",
        title: "Return computed edges",
        program: "[Y, E] = discretize([0 1], 2)",
        display_output: Some("Y = [1 2], E = [0 0.5 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Y, [1 2]));\nassert(isequal(E, [0 0.5 1]));" },
    },
    BuiltinExample {
        id: "typed-replacements",
        title: "Preserve integer replacement values",
        program: "labels = uint64([0x0020000000000001u64 0xFFFFFFFFFFFFFFFFu64]);\nY = discretize([-1 0.5 1.5 3], [0 1 2], labels)",
        display_output: Some("Y is uint64 and uses zero outside the bins"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Y, \"uint64\"));\nassert(isequal(Y, uint64([0 labels(1) labels(2) 0])));" },
    },
    BuiltinExample {
        id: "infinite-outer-edges",
        title: "Bin infinite values with infinite outer edges",
        program: "Y = discretize([-Inf -1 1 Inf NaN], [-Inf 0 Inf])",
        display_output: Some("Y = [1 1 2 2 NaN]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequaln(Y, [1 1 2 2 NaN]));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What happens at the final edge?", answer: "With the default left-edge convention, the final bin includes its right edge. With right-edge inclusion, the first bin includes its left edge." },
    BuiltinDocumentationFaq { question: "Does an integer replacement vector remain integer?", answer: "Yes. RunMat preserves its fixed-width class and uses zero for elements that do not belong to a bin." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "histcounts",
        target: BuiltinDocumentationLinkTarget::Builtin("histcounts"),
    },
    BuiltinDocumentationLink {
        label: "Compatible discretize reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/double.discretize.html",
        ),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Numeric binning runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/binning/discretize/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Edges, output arity, integer exactness, replacements, missing values, and provider behavior", location: "crates/runmat-runtime/src/builtins/array/binning/discretize/tests" }],
    notes: &["The documentation states unsupported compatible forms explicitly; they remain tracked capability gaps rather than inferred support."],
};

pub(super) const DISCRETIZE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("discretize"),
    slug: Some("discretize"),
    summary: "Assign real numeric values to bins or typed replacement values.",
    description: "`discretize` maps each element of a real numeric or logical array to an explicit or computed numeric bin.",
    keywords: &["discretize", "bins", "edges", "binning", "grouping"],
    related: &["histcounts", "histogram"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("R2015a"),
    status: Some(BuiltinDocumentationStatus::Stable),
};

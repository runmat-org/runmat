use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("groupsummary"),
    slug: Some("groupsummary"),
    summary: "Compute summary statistics by group for table or array data.",
    description: "`groupsummary` groups table rows or array observations and computes per-group statistics. Table input returns a table; array input returns summary values and can additionally return grouping values and counts.",
    keywords: &["groupsummary", "group", "table", "array", "statistics", "groupbins"],
    related: &["groupcounts", "findgroups", "splitapply"],
    sections: &[
        BuiltinDocumentationSection { heading: "Table and array forms", paragraphs: &[
            "Table forms select grouping and data variables and implement mean, sum, median, min, max, count, and numel methods.",
            "Array forms accept numeric vectors and matrices or cell arrays of numeric arrays. Matrix columns are independent variables. `[B,BG,BC]` returns summary values, original-class group values when representable, and double group counts.",
            "Integer grouping keys, integer extrema, and array/table reconstruction use typed storage. Wide integer values that cannot cross a floating-point summary boundary exactly reject instead of silently rounding.",
        ] },
        BuiltinDocumentationSection { heading: "Bins and options", paragraphs: &[
            "One numeric grouping variable supports edge vectors and positive bin counts. `IncludedEdge`, `IncludeMissingGroups`, and a supported subset of `IncludeEmptyGroups` are available. Native typed-integer bin-count and zero/one controls are RunMat-only forms; scalar-double and logical controls work in MATLAB-compatible mode.",
            "Multi-group bin specifications, time bins, additional named methods, function-handle methods, and some table selector forms are not implemented. `IncludeEmptyGroups=true` for categorical level expansion rejects rather than claiming to enumerate absent levels.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Grouping runs on the host and is a fusion boundary. Automatic internal residency can gather before grouping; explicitly resident interactive GPU data is a mode-gated RunMat extension.",
            "The runtime grouping keys and table schema are not statically known. Static analysis distinguishes known table input from array input but does not invent group counts, output dimensions, or a table variable schema.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "array-mean-and-counts", title: "Summarize an array and return groups", program: "[B,BG,BC] = groupsummary([10;20;30],[1;1;2],'mean');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B,[15;30])); assert(isequal(BG,[1;2])); assert(isequal(BC,[2;1]));" } },
        BuiltinExample { id: "wide-integer-extrema", title: "Preserve wide integer groups", program: "base = uint64(9007199254740992); [B,BG] = groupsummary([base+3;base+1;base+4],[base;base+1;base],'max');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(B,'uint64')); assert(isa(BG,'uint64')); assert(isequal(B,[base+4;base+1])); assert(isequal(BG,[base;base+1]));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "groupcounts", target: BuiltinDocumentationLinkTarget::Builtin("groupcounts") },
        BuiltinDocumentationLink { label: "findgroups", target: BuiltinDocumentationLinkTarget::Builtin("findgroups") },
        BuiltinDocumentationLink { label: "splitapply", target: BuiltinDocumentationLinkTarget::Builtin("splitapply") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/analytics.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Table and array grouping, missing controls, bins, and integer precision", location: "crates/runmat-runtime/src/builtins/table/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM integer statistical semantics", location: "crates/runmat-vm/tests/integer_statistical_semantics.rs" },
        ],
        notes: &["Legacy source fields, behavior, links, and integer capability details are retained; examples now carry assertions.", "Legacy gpu_support and fusion map to typed host placement. requires_feature was null; tested maps to verification evidence."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

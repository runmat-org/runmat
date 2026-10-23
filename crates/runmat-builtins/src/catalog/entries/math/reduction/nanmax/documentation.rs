use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nanmax"), slug: Some("nanmax"),
    summary: "Return maximum values while omitting NaNs.",
    description: "`nanmax` is a compatibility form for maximum reductions and pairwise comparisons that ignore `NaN` values. It maps to `max` with `\"omitnan\"` semantics.",
    keywords: &["nanmax", "max", "maximum", "omitnan", "statistics"],
    related: &["max", "min", "mean"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`nanmax(X)` reduces along the first non-singleton dimension and skips `NaN` values in each slice.",
            "`nanmax(X, [], dim)` and `nanmax(X, [], vecdim)` reduce along the specified dimension or dimension vector.",
            "`nanmax(X, [], 'all')` collapses every element into a scalar; `nanmax(X, [], 'linear')` also returns linear indices when two outputs are requested.",
            "`nanmax(..., 'ComparisonMethod', method)` supports the same real and complex comparison modes as `max` while still omitting NaNs.",
            "`nanmax(A, B)` performs elementwise maximum with MATLAB implicit expansion and chooses the non-NaN side when exactly one operand is `NaN`.",
            "If every value considered for a result is `NaN`, the returned value is `NaN`.",
            "Two-output reduction calls return one-based maximum indices; pairwise calls support one output only, as with max.",
        ]},
        BuiltinDocumentationSection { heading: "Forms, classes, and shapes", paragraphs: &[
            concat!(
            "Reduction parsing, comparison modes, implicit expansion, integer storage, and index selection share max's implementation. NaN omission is applied as a",
            " typed execution policy after validating options, so an explicit include flag cannot override nanmax. All-NaN slices return NaN; ties select the first",
            " matching index.",
        ),
            concat!(
            "Reduction calls accept zero through two requested outputs; pairwise forms accept zero or one. Indices are double and one-based. Typed integer data, pa",
            "irwise operands, or dimensions require the named RunMat extension. Sparse and other unsupported max representations are rejected rather than densified",
            " by nanmax.",
        ),
        ]},
        BuiltinDocumentationSection { heading: "Execution and allocation", paragraphs: &[
            concat!(
            "Inputs and controls retain their invocation owners across suspension. Each call uses its selected execution context and cancellation authority. Provid",
            "er outputs are checked for owner, device, physical class, shape, and forbidden input aliasing. Operational errors, cancellation, authority loss, and i",
            "nvalid attempted outputs remain terminal; no host replay follows them.",
        ),
            concat!(
            "NaN-omitting maxima are a declared host-result boundary: resident inputs are read through their exact owner, and results stay on the host. No omit-NaN",
            " provider reduction is claimed.",
        ),
            concat!(
            "Finite host allocation is not yet implemented for this operation's array scratch, shape, and output collections. Such paths reject a Managed allocatio",
            "n policy before unowned work; ordinary Unbounded host execution remains available. This restriction is an unfinished allocation capability, not a prom",
            "ise of a bounded host implementation. Native/browser execution shares the Runtime binding; long synchronous host loops do not promise browser yielding",
            ".",
        ),
            concat!(
            "These operations execute through their Runtime bindings, not a certified fused reduction. The live planner declines fused prod and nnz execution. Dist",
            "ributed execution is not declared by these entries.",
        ),
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Column-wise maxima ignoring NaNs",
            program: "A = [NaN 4 2; 3 NaN NaN];\nM = nanmax(A)",
            display_output: Some("M = [3 4 2]"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(M,[3 4 2]));" },
        },
        BuiltinExample {
            id: "example-2", title: "Pairwise maximum ignoring NaNs",
            program: "P = nanmax([NaN 2 NaN], [3 NaN NaN])",
            display_output: Some("P = [3 2 NaN]"),
            compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(P(1)==3 && P(2)==2 && isnan(P(3)));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Should new code use `nanmax` or `max(..., 'omitnan')`?", answer: "Use `max(..., 'omitnan')` for new MATLAB-style code. `nanmax` is provided for compatibility with existing codebases." },
        BuiltinDocumentationFaq { question: "Does `nanmax` support pairwise array inputs?", answer: "Yes. nanmax(A,B) performs implicit expansion and chooses the non-NaN operand when only one side is NaN. Pairwise calls return one value output and reject a second output." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "max", target: BuiltinDocumentationLinkTarget::Builtin("max") },
        BuiltinDocumentationLink { label: "min", target: BuiltinDocumentationLinkTarget::Builtin("min") },
        BuiltinDocumentationLink { label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean") },
        BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/math/reduction/max.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/max.rs") },
        BuiltinDocumentationLink { label: "MATLAB reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/stats/nanmax.html") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/math/reduction/max.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/max.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::math::reduction::max::tests", location: "builtins::math::reduction::max::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "crates/runmat-vm/tests/basics.rs", location: "crates/runmat-vm/tests/basics.rs" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

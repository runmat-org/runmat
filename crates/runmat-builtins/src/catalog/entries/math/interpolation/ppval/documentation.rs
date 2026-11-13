use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ppval"),
    slug: Some("ppval"),
    summary: "Evaluate piecewise-polynomial structures in MATLAB and RunMat.",
    description: "`ppval(pp, xq)` evaluates a piecewise-polynomial structure at query points. The structure is typically returned by `spline` or `pchip`.",
    keywords: &["ppval", "piecewise polynomial", "spline", "pchip"],
    related: &["spline", "pchip"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Syntax",
            paragraphs: &["`yq = ppval(pp, xq)` evaluates at scalar or array query points. A single-valued polynomial preserves the query shape; a polynomial with multiple component values places its component dimensions before the query dimensions.",
                "`pp` contains `form`, `breaks`, `coefs`, `pieces`, `order`, and `dim` fields. In a vector-valued pp, flattened coefficient rows place component values within each piece before moving to the next piece. The coefficient table is evaluated with Horner's method in the selected interval. Queries outside the breakpoint range use the first or last polynomial piece for extrapolation."],
        },
        BuiltinDocumentationSection {
            heading: "Execution",
            paragraphs: &["Typed-integer query points are a RunMat-mode extension. They must be exactly representable in double before evaluation.",
                "Resident query points are downloaded to the CPU reference implementation. This is an intentional materialization boundary, including for explicit device inputs; `ppval` does not promise a resident output or a device kernel. The returned numerical values are double."],
        },
    ],
    examples: &[BuiltinExample {
        id: "evaluate-spline-pp",
        title: "Evaluate a spline pp structure",
        program: "pp = spline([1 2 3], [1 4 9]);\nyq = ppval(pp, [1.5 2.5])",
        display_output: Some("yq = [2.25 6.25]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(numel(yq)==2); assert(abs(yq(1)-2.25)<1e-10); assert(abs(yq(2)-6.25)<1e-10); assert(strcmp(class(yq),'double'));",
        },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "spline", target: BuiltinDocumentationLinkTarget::Builtin("spline") },
        BuiltinDocumentationLink { label: "pchip", target: BuiltinDocumentationLinkTarget::Builtin("pchip") },
        BuiltinDocumentationLink { label: "MATLAB ppval reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/ppval.html") },
        BuiltinDocumentationLink { label: "MATLAB mkpp coefficient layout", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/mkpp.html") },
        BuiltinDocumentationLink { label: "Open an issue", target: BuiltinDocumentationLinkTarget::External("https://github.com/runmat-org/runmat/issues/new/choose") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "ppval runtime entry", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/ppval.rs") },
            BuiltinDocumentationLink { label: "Piecewise-polynomial structure and coefficient layout", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/format.rs") },
            BuiltinDocumentationLink { label: "Piecewise-polynomial evaluation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/evaluation.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Piecewise-polynomial evaluation and diagnostics", location: "builtins::math::interpolation::ppval::tests" }],
        notes: &["The legacy GPU precision listing described admitted resident input classes, not a device ppval kernel. The CPU evaluator publishes double values."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

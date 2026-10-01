use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pchip"),
    slug: Some("pchip"),
    summary: "Construct or evaluate shape-preserving piecewise cubic Hermite interpolants.",
    description: "`pchip(x,y)` builds a piecewise-polynomial structure; `pchip(x,y,xq)` evaluates it. The Hermite slopes preserve monotonic sample intervals without the overshoot of an unconstrained cubic spline.",
    keywords: &["pchip", "piecewise cubic hermite", "shape preserving", "interpolation"],
    related: &["interp1", "ppval", "spline"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Syntax and shape",
            paragraphs: &[
                "`pp = pchip(x,y)` returns a structure with `form`, `breaks`, `coefs`, `pieces`, `order`, and `dim`; it can be evaluated with `ppval`. `Vq = pchip(x,y,xq)` evaluates directly and extrapolates outside the sample interval.",
                "`x` is a strictly increasing vector. For array-valued `y`, samples lie on its last axis; the other axes describe function components. Those component axes precede the query axes in `Vq`. RunMat also continues to accept its earlier unambiguous first-axis array form; that is a RunMat compatibility form, not a MATLAB-parity claim.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "The CPU implementation constructs double-precision coefficients and returns double numerical values. MATLAB documents single and double inputs. Typed integer sample locations, values, and query points are separate RunMat-mode extensions; each must convert to binary64 exactly.",
                "Resident inputs are intentionally gathered to the CPU reference implementation. This is an admitted materialization boundary, including for explicitly device-resident data; no GPU PCHIP kernel or resident output is promised. The operation is a fusion boundary.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "evaluate-shape-preserving-cubic",
            title: "Interpolate monotone samples",
            program: "x = [1 2 3];\ny = [1 4 9];\nyq = pchip(x, y, [1.5 2.5])",
            display_output: Some("yq = [2.28125 6.21875]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(yq)==2); assert(abs(yq(1)-2.28125)<1e-10); assert(abs(yq(2)-6.21875)<1e-10); assert(strcmp(class(yq),'double'));",
            },
        },
        BuiltinExample {
            id: "construct-piecewise-polynomial",
            title: "Construct and evaluate a pp structure",
            program: "pp = pchip([1 2 3], [1 4 9]);\nyq = ppval(pp, [1.5 2.5])",
            display_output: Some("yq = [2.28125 6.21875]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(pp.form,'pp')); assert(pp.pieces==2); assert(abs(yq(1)-2.28125)<1e-10); assert(abs(yq(2)-6.21875)<1e-10);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "interp1", target: BuiltinDocumentationLinkTarget::Builtin("interp1") },
        BuiltinDocumentationLink { label: "ppval", target: BuiltinDocumentationLinkTarget::Builtin("ppval") },
        BuiltinDocumentationLink { label: "MATLAB pchip reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/pchip.html") },
        BuiltinDocumentationLink { label: "Open an issue", target: BuiltinDocumentationLinkTarget::External("https://github.com/runmat-org/runmat/issues/new/choose") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "PCHIP runtime entry", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pchip.rs") },
            BuiltinDocumentationLink { label: "Sample-series axes and component layout", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/series.rs") },
            BuiltinDocumentationLink { label: "PCHIP coefficients", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/coefficients.rs") },
            BuiltinDocumentationLink { label: "Piecewise-polynomial structure", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/format.rs") },
            BuiltinDocumentationLink { label: "Piecewise-polynomial evaluation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/evaluation.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "PCHIP direct forms and diagnostics", location: "builtins::math::interpolation::pchip::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Typed integer interpolation semantics", location: "crates/runmat-vm/tests/integer_numeric_transform_semantics.rs" },
        ],
        notes: &["The previous editable JSON example's first displayed value, 2.21875, disagreed with the PCHIP slopes for its own source data. The canonical example asserts the actual 2.28125 value. The prior GPU precision listing described admitted inputs, not a device kernel."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

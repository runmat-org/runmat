use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("spline"),
    slug: Some("spline"),
    summary: "Construct cubic spline interpolants or evaluate them at query points.",
    description: "`spline(x,y)` constructs a cubic piecewise-polynomial structure, and `spline(x,y,xq)` evaluates it. The default end condition is not-a-knot; two extra values along Y's sample axis select specified endpoint slopes.",
    keywords: &["spline", "cubic spline", "interpolation", "pp", "ppval"],
    related: &["interp1", "ppval", "pchip"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Syntax and end conditions",
            paragraphs: &[
                "`pp = spline(x,y)` returns a structure with `form`, `breaks`, `coefs`, `pieces`, `order`, and `dim`; `Vq = spline(x,y,xq)` evaluates directly. Direct evaluation extrapolates outside the sample interval using the endpoint polynomial pieces.",
                "`x` must be a strictly increasing vector. With the same number of Y samples, spline uses not-a-knot end conditions (linear with two points, quadratic with three). If Y has two more values than X on its last sample axis, the first and last Y values specify endpoint slopes and the interior values are samples. This also applies independently to every component of matrix- or array-valued Y.",
                "For array-valued Y, the final axis contains samples. Its other axes precede query axes in the output. RunMat also retains an earlier unambiguous first-axis array form as a RunMat compatibility behavior, not a MATLAB-parity claim.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "The CPU reference implementation stores double coefficients and returns double numerical values. MATLAB documents single and double input classes. Typed integer sample locations, sample values, and query points are independent RunMat-mode extensions; each must be exactly representable in binary64 before interpolation.",
                "Resident inputs are intentionally gathered for CPU coefficient construction and evaluation. This includes explicit device inputs: spline has no resident GPU kernel or GPU-output promise, and is a fusion boundary.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "construct-and-evaluate-spline",
            title: "Build a spline pp structure",
            program: "x = [1 2 3];\ny = [1 4 9];\npp = spline(x, y);\nyq = ppval(pp, [1.5 2.5])",
            display_output: Some("yq = [2.25 6.25]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(pp.form,'pp')); assert(pp.pieces==2); assert(abs(yq(1)-2.25)<1e-10); assert(abs(yq(2)-6.25)<1e-10);",
            },
        },
        BuiltinExample {
            id: "evaluate-spline-directly",
            title: "Evaluate a spline directly",
            program: "x = [1 2 3];\ny = [1 4 9];\nyq = spline(x, y, 1.5)",
            display_output: Some("yq = 2.25"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(abs(yq-2.25)<1e-10); assert(strcmp(class(yq),'double'));",
            },
        },
        BuiltinExample {
            id: "clamp-endpoint-slopes",
            title: "Specify endpoint slopes",
            program: "x = [0 1 2];\npp = spline(x, [0 0 1 8 12]);\nyq = ppval(pp, [0.5 1.5])",
            display_output: Some("yq = [0.125 3.375]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(abs(yq(1)-0.125)<1e-10); assert(abs(yq(2)-3.375)<1e-10); assert(pp.pieces==2);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "interp1", target: BuiltinDocumentationLinkTarget::Builtin("interp1") },
        BuiltinDocumentationLink { label: "ppval", target: BuiltinDocumentationLinkTarget::Builtin("ppval") },
        BuiltinDocumentationLink { label: "MATLAB spline reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/spline.html") },
        BuiltinDocumentationLink { label: "Open an issue", target: BuiltinDocumentationLinkTarget::External("https://github.com/runmat-org/runmat/issues/new/choose") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Spline runtime entry", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/spline.rs") },
            BuiltinDocumentationLink { label: "Sample-series axes and component layout", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/series.rs") },
            BuiltinDocumentationLink { label: "Spline coefficients and end conditions", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/coefficients.rs") },
            BuiltinDocumentationLink { label: "Piecewise-polynomial structure", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/format.rs") },
            BuiltinDocumentationLink { label: "Piecewise-polynomial evaluation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/pp/evaluation.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Spline direct, pp, endpoint-slope, and array geometry cases", location: "builtins::math::interpolation::spline::tests" }],
        notes: &["The legacy GPU precision listing described accepted resident input classes, not a GPU spline kernel. Exact output class behavior is documented as RunMat behavior without asserting unverified MATLAB parity."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

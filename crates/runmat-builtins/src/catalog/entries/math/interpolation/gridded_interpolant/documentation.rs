use crate::*;

const IMPLEMENTATION: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "griddedInterpolant runtime",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/gridded_interpolant.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("griddedInterpolant"),
    slug: Some("griddedInterpolant"),
    summary: "Create an object that interpolates real dense data on a rectilinear grid.",
    description: "`griddedInterpolant` constructs an interpolant object with GridVectors, Values, Method, and ExtrapolationMethod properties. Calling the object with parentheses evaluates it at query points.",
    keywords: &["griddedInterpolant", "interpolation", "grid", "linear", "nearest", "pchip", "spline", "makima"],
    related: &["interp1", "interp2", "pchip", "spline"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Construction",
            paragraphs: &[
                "`griddedInterpolant(V)` uses default grid vectors `1:size(V,dim)`. `griddedInterpolant(x,v)` and `griddedInterpolant({x1,...,xn},V)` use explicit compact grid vectors. Full-grid `griddedInterpolant(X1,...,Xn,V)` extracts vectors from rectilinear monotone grids.",
                "Grid vectors are finite, strictly increasing real vectors with at least two points. Values are real dense arrays whose leading dimensions match the grid. Single-precision Values remain single in the object and in tensor evaluation; scalar double results use numeric scalars.",
                "Method may be linear, nearest, next, previous, pchip, spline, or makima. Pchip and makima are 1-D; spline uses shared not-a-knot coefficients for 1-D and a tensor product for 2-D. ExtrapolationMethod also accepts none. Cubic convolution, multidimensional makima, and spline above 2-D return explicit errors rather than a substitute interpolation. Makima extrapolation currently requires makima interpolation.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Evaluation and placement",
            paragraphs: &[
                "`F(xq)` evaluates a 1-D interpolant. `F(xq1,...,xqn)` accepts matching-shape query arrays, `F(Xq)` accepts row-wise N-D query points, and `F({xgq1,...,xgqn})` forms a full query grid. The same object indexing path exposes GridVectors, Values, Method, and ExtrapolationMethod.",
                "Typed integer sample grids, values, and queries require exact binary64 conversion and are RunMat-mode extensions. The object and evaluation are host-side; supported resident inputs gather to that path after compatibility checks. No griddedInterpolant device kernel or fusion path is claimed.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "one-dimensional-grid",
            title: "Interpolate explicit sample points",
            program: "F = griddedInterpolant([0 1 2],[0 10 40]); yq = F([0.5 1.5])",
            display_output: Some("yq = [5 25]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(numel(yq)==2); assert(abs(yq(1)-5)<1e-12); assert(abs(yq(2)-25)<1e-12);" },
        },
        BuiltinExample {
            id: "two-dimensional-grid",
            title: "Interpolate from two grid vectors",
            program: "F = griddedInterpolant({[0 1],[0 2]},[0 20; 10 30]); yq = F(0.5,1)",
            display_output: Some("yq = 15"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(yq-15)<1e-12);" },
        },
        BuiltinExample {
            id: "one-dimensional-makima",
            title: "Evaluate modified Akima interpolation",
            program: "F = griddedInterpolant([0 1 2 3 4],[0 0 0 1 0],'makima'); yq = F(2.5)",
            display_output: Some("yq = 0.443181818181818"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(yq-39/88)<1e-12);" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "interp1", target: BuiltinDocumentationLinkTarget::Builtin("interp1") },
        BuiltinDocumentationLink { label: "interp2", target: BuiltinDocumentationLinkTarget::Builtin("interp2") },
        BuiltinDocumentationLink { label: "pchip", target: BuiltinDocumentationLinkTarget::Builtin("pchip") },
        BuiltinDocumentationLink { label: "spline", target: BuiltinDocumentationLinkTarget::Builtin("spline") },
        BuiltinDocumentationLink { label: "MATLAB griddedInterpolant reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/griddedinterpolant.html") },
        BuiltinDocumentationLink { label: "Open an issue", target: BuiltinDocumentationLinkTarget::External("https://github.com/runmat-org/runmat/issues/new/choose") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[IMPLEMENTATION],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "construction, queries, typed classes, and diagnostics", location: "builtins::math::interpolation::gridded_interpolant::tests" }],
        notes: &["The legacy sidecar's GPU and fusion fields describe the current host object route, not a provider kernel; its syntax, behaviors, examples, links, and source are retained here."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub(super) const SUBSREF_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("griddedInterpolant.subsref"),
    slug: Some("griddedInterpolant.subsref"),
    summary: "Read an interpolant property or evaluate query points through object indexing.",
    description: "The method-only subsref binding implements parenthesis evaluation and public property reads for griddedInterpolant objects. The standard substruct path determines the result, so static output kind and shape remain dynamic until the path is known.",
    keywords: &["griddedInterpolant", "subsref", "indexing", "interpolation"],
    related: &["griddedInterpolant"],
    sections: &[BuiltinDocumentationSection {
        heading: "Object indexing",
        paragraphs: &["Parentheses dispatch to interpolant evaluation. Dot indexing reads GridVectors, Values, Method, or ExtrapolationMethod. Brace indexing is unsupported. Resident numeric query values are gathered to the host implementation after typed extension checks."],
    }],
    examples: &[BuiltinExample {
        id: "parenthesis-evaluation",
        title: "Evaluate through the object indexing method",
        program: "F = griddedInterpolant([0 1 2],[0 10 40]); yq = F([0.5 1.5])",
        display_output: Some("yq = [5 25]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(numel(yq)==2); assert(abs(yq(1)-5)<1e-12); assert(abs(yq(2)-25)<1e-12);" },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "griddedInterpolant", target: BuiltinDocumentationLinkTarget::Builtin("griddedInterpolant") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[IMPLEMENTATION],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "griddedInterpolant subsref and property tests", location: "builtins::math::interpolation::gridded_interpolant::tests" }],
        notes: &["The user-facing documentation remains under the constructor; this identity is the method-only object indexing binding."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

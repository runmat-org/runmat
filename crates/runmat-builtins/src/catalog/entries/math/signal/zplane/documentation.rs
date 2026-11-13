use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("zplane"),
    slug: Some("zplane"),
    summary: "Plot zeros and poles of a digital filter.",
    description: "`zplane` draws the unit circle, real and imaginary axes, and markers for filter zeros and poles. Row-vector inputs are interpreted as transfer-function coefficients `b` and `a`; column-vector inputs are interpreted as explicit zero and pole locations.",
    keywords: &["zplane", "zero pole plot", "digital filter", "signal processing", "roots"],
    related: &["roots", "freqz", "filter", "plot"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`zplane(b, a)` computes `roots(b)` and `roots(a)` and plots the resulting zeros and poles.",
            "`zplane(z, p, k)` accepts explicit zero and pole locations; the gain is validated for compatibility and does not affect marker placement.",
            "`zplane(sos)` accepts a second-order-section matrix with six columns `[b0 b1 b2 a0 a1 a2]` and plots all section roots.",
            "A leading axes handle selects the target plot for each form: `zplane(ax,sos)`, `zplane(ax,b,a)`, or `zplane(ax,z,p,k)`. Assigned forms return handles from that target.",
            "The plot uses equal axis scaling, grid lines, the unit circle, and symmetric limits large enough to include all finite zeros and poles.",
            "When assigned, `h = zplane(...)` returns a row vector of graphics handles for the guide lines and marker plots.",
            "The numeric input contract is single or double. RunMat mode additionally accepts native integer coefficients, locations, SOS entries, and gains through a checked conversion into the floating root and graphics domain.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &[
            "Host root extraction and plotting gather resident coefficient inputs at the documented materialization boundary after known SOS, vector, and gain metadata are validated. Graphics handles remain host values; no GPU root or rendering kernel is claimed.",
            "No-output calls still render the plot. Assigned calls return a row of graphics handles for guides and present zero/pole marker plots. Plotting requires a graphics host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Plot zeros and poles of a simple IIR filter", program: "b = 1;\na = [1 -0.8];\nzplane(b, a);\ntitle('Zero-Pole Plot');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(findobj(gcf,'Type','axes'))>=1);" } },
        BuiltinExample { id: "example-2", title: "Use explicit zero and pole vectors", program: "z = [0.2; -0.2];\np = [0.8; 0.6];\nzplane(z, p, 1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(findobj(gcf,'Type','axes'))>=1);" } },
        BuiltinExample { id: "example-3", title: "Plot roots from second-order sections", program: "sos = [1 0 0 1 -0.8 0];\nzplane(sos);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(findobj(gcf,'Type','axes'))>=1);" } },
        BuiltinExample { id: "target-axes", title: "Render into selected axes", program: "f = figure(); ax = axes('Parent',f); h = zplane(ax, [1], [1 -0.8]);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(h)==4); assert(numel(findobj(ax,'Type','line'))>=1);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How does RunMat distinguish coefficients from explicit zeros and poles?", answer: "Row-vector two-argument inputs are treated as transfer-function coefficients. Column-vector two-argument inputs are treated as explicit zero and pole locations, matching the common MATLAB zplane convention." },
        BuiltinDocumentationFaq { question: "Does zplane compute the frequency response?", answer: "No. `zplane` only computes roots and renders the zero-pole diagram. Use `freqz` to compute or plot frequency response values." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "freqz", target: BuiltinDocumentationLinkTarget::Builtin("freqz") },
        BuiltinDocumentationLink { label: "filter", target: BuiltinDocumentationLinkTarget::Builtin("filter") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/zplane.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Input and root interpretation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/zplane.rs") },
            BuiltinDocumentationLink { label: "Plot presentation and handles", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/zplane/presentation.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Input, output and failure semantics", location: "builtins::math::signal::zplane::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xlim"),
    slug: Some("xlim"),
    summary: "Query or set x-axis limits.",
    description: "`xlim([xmin xmax])` sets the x-axis limits, while `xlim()` returns the current limits as a double vector.",
    keywords: &["xlim", "axis limits", "plotting"],
    related: &["ylim", "axis"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Behavior and compatibility", paragraphs: &[
            "The two-element limits vector may use single, double, or any native integer class.",
            "Native integer endpoints are ordered using their exact values before conversion to graphics coordinates.",
            "Endpoints must be finite, increasing, and remain distinct in the graphics coordinate domain.",
            "The accepted text tokens include `'auto'`, `'tight'`, and `'manual'`: `'auto'` restores automatic selection, `'tight'` selects the tight automatic-limit method, and `'manual'` freezes the current numeric range. `'mode'` queries the auto/manual state.",
            "`'tight'`, `'padded'`, and `'tickaligned'` select distinct automatic limit methods; `'method'` queries the selected method.",
            "An existing axes handle can be supplied first for query or setter forms without selecting it as the current axes. Semiautomatic infinite endpoints and datetime/duration/categorical limits still require graphics-state support.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "The control updates host graphics state. Resident control operands reject before provider access; existing resident plot data retains its rendering owner.",
            "The whole-call allocation, graphics presentation, and callback execution routes still require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Set integer limits", program: "plot(1:10);\nxlim(int16([0 12]))", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(xlim(),[0 12]));" } },
        BuiltinExample { id: "example-2", title: "Query current limits", program: "limits = xlim()", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(limits,'double')); assert(isequal(size(limits),[1 2]));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "ylim", target: BuiltinDocumentationLinkTarget::Builtin("ylim") },
        BuiltinDocumentationLink { label: "axis", target: BuiltinDocumentationLinkTarget::Builtin("axis") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/xlim.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "xlim state and validation tests", location: "builtins::plotting::xlim::tests" }],
        notes: &["Legacy behavior, examples, and links are retained. Native and browser graphics execution remains subject to integration gates."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

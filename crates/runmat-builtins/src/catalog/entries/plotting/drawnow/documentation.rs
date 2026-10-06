use crate::*;

const GRAPHICS: BuiltinExampleRequirements = BuiltinExampleRequirements {
    host: BuiltinExampleHostRequirement::Any,
    engine: BuiltinExampleEngine::Default,
    compiler: &[],
    runtime: &[BuiltinRuntimeCapability::PlotCore],
    toolchain: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("drawnow"),
    slug: Some("drawnow"),
    summary: "Present pending figure updates and process graphics callbacks.",
    description: "`drawnow` asks the selected graphics host to present pending figure updates and process callbacks. Its control forms can rate-limit presentation or defer callbacks without discarding them.",
    keywords: &["drawnow", "graphics", "render", "callbacks", "limitrate"],
    related: &["figure", "plot", "pause", "animatedline"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Forms",
            paragraphs: &[
                "`drawnow` requests an immediate presentation and processes queued graphics callbacks. It does not create a figure when none exists; callback processing is a separate host obligation even in that case.",
                "`drawnow limitrate` skips presentation when the renderer is busy or fewer than 50 ms have elapsed since its last presentation, but still processes callbacks. `drawnow nocallbacks` presents without processing callbacks; those callbacks remain queued for a later full flush. The two controls can be combined as `drawnow limitrate nocallbacks`.",
                "Legacy `drawnow update` skips presentation only while busy and defers callbacks; it is not a 50 ms limit-rate request. Legacy `drawnow expose` presents immediately and defers callbacks. Both forms remain supported but are not recommended for new code.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Host and output behavior",
            paragraphs: &[
                "The active native, browser, or Desktop presentation service owns the window/canvas receipt and callback queue. A numeric figure handle alone does not grant host or session authority. A host without a graphics presentation service fails with a typed capability error rather than reporting a successful no-op.",
                "MATLAB-compatible calls are statements with no output. Requesting a logical completion status is a RunMat-only extension. A rate-limited call may complete without presenting a new frame; its status reports host completion, not proof of new pixels.",
                "The call can suspend. Cancellation before dispatch has no host effect; once a host request may have committed, a lost acknowledgement is indeterminate rather than falsely reported as unpublished.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "flush-plot",
            title: "Present a changed plot",
            program: "x = 0:0.1:1;\nplot(x,x.^2);\ndrawnow;",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: GRAPHICS,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(ishandle(gcf));" },
        },
        BuiltinExample {
            id: "limited-animation",
            title: "Limit presentation while updating an animated line",
            program: "h = animatedline;\nfor k = 1:10\n    addpoints(h,k,sin(k));\n    drawnow limitrate;\nend\ndrawnow;",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: GRAPHICS,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(ishandle(gcf));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does drawnow create a figure?", answer: "No. It can still process queued callbacks when there is no figure to present." },
        BuiltinDocumentationFaq { question: "Does limitrate discard callbacks?", answer: "No. It can defer a frame, but callbacks are processed unless nocallbacks is also requested." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "figure", target: BuiltinDocumentationLinkTarget::Builtin("figure") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "Graphics host integration", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/host-integration") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Drawnow runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/drawnow.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Drawnow controls and inference", location: "catalog::entries::plotting::drawnow::tests" }],
        notes: &["The canonical controls are source-defined; native GUI, browser frame, Desktop callback, and cancellation execution remain qualification gates."],
    },
    introduced: None,
    status: None,
};

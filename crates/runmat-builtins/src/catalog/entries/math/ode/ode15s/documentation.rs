use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ode15s"),
    slug: Some("ode15s"),
    summary: "Solve stiff ODE systems using adaptive implicit integration with MATLAB-compatible `ode15s` call forms.",
    description: "`ode15s` integrates stiff first-order ODE systems `y' = f(t, y)` over `tspan` and returns time/state outputs in the currently supported two-output time/state arrays.",
    keywords: &["ode15s", "ode", "stiff", "implicit", "adaptive step"],
    related: &["ode23", "ode45"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Supports scalar, vector, and matrix-valued initial conditions.",
            "Accepts options as a struct, including `RelTol`, `AbsTol`, `InitialStep`, and `MaxStep`.",
            "The documented tspan, y0, and derivative domains are single or double. Native integer and logical solver values are independently gated RunMat extensions, and every integer-to-double solver boundary requires exact representability.",
            "Typed integer `MaxSteps` is parsed as an exact structural count; integer floating controls are checked before conversion.",
            "The current RunMat integration loop and outputs are double even when all floating inputs are single; native-single output propagation remains a general compatibility gap rather than an integer extension.",
            "Event outputs and solution-structure forms are not implemented by the current core solver surface.",
            "When `tspan` has more than two points, the solver returns values at each requested time.",
            "With one output, returns only the `y` trajectory matrix."
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and compatibility",
            paragraphs: &[
                "The integration loop runs on the CPU and invokes the supplied derivative through RunMat's ordinary callback service. Automatic resident input may gather; explicit gpuArray fallback is a gated RunMat extension.",
                "The current single-output form returns the y trajectory rather than MATLAB's solution structure. Event arguments and event outputs are not yet supported. These are interface gaps, not alternate MATLAB-compatible forms.",
                "The current solver returns double time and state arrays, including for single initial conditions. Integer and logical inputs are separately gated RunMat extensions.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "example-1",
        title: "Integrate a stiff scalar decay system",
        program: "[t, y] = ode15s(@(t, y) -15*y, [0 1], 1)",
        display_output: Some("y(end) is small and positive"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(size(y,2)==1); assert(size(y,1)==size(t,1)); assert(abs(y(1)-1)<1e-12);",
        },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "ode23",
            target: BuiltinDocumentationLinkTarget::Builtin("ode23"),
        },
        BuiltinDocumentationLink {
            label: "ode45",
            target: BuiltinDocumentationLinkTarget::Builtin("ode45"),
        },
        BuiltinDocumentationLink {
            label: "Source",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/ode/ode15s.rs"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "ODE solver unit tests",
            location: "builtins::math::ode::ode15s::tests",
        }],
        notes: &[
            "The legacy sidecar's summary, keywords, behaviors, example, links, GPU and fusion claims were reconciled. Standalone example acceptance and hardware qualification remain separate gates.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ode45"),
    slug: Some("ode45"),
    summary: "Solve nonstiff ODE systems using adaptive Dormand-Prince 5(4) integration.",
    description: "`ode45` integrates nonstiff first-order ODE systems over `tspan` using adaptive RK45 stepping and returns time/state output arrays.",
    keywords: &["ode45", "ode", "nonstiff", "Dormand-Prince", "adaptive step"],
    related: &["ode15s", "ode23"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Supports scalar, vector, and matrix-valued initial conditions.",
            "Accepts options as a struct, including `RelTol`, `AbsTol`, `InitialStep`, `MaxStep`, and the positive-integer output refinement factor `Refine`. Empty numeric fields select defaults.",
            "The documented tspan, y0, and derivative domains are single or double. Native integer and logical solver values are independently gated RunMat extensions, and every integer-to-double solver boundary requires exact representability.",
            "Typed integer `MaxSteps` and `Refine` are parsed as exact structural counts; integer floating controls are checked before conversion.",
            "The current RunMat integration loop and outputs are double even when all floating inputs are single; native-single output propagation remains a general compatibility gap rather than an integer extension.",
            "Event outputs and solution-structure forms are not implemented by the current core solver surface.",
            "When `tspan` has more than two points, the solver returns values at each requested time.",
            "Two-endpoint time spans use Refine=4 by default; Refine=1 returns accepted-step endpoints only. Intermediate samples use the Dormand-Prince/Shampine quartic continuous extension without additional derivative calls.",
            "Refine does not change explicit time grids with more than two nodes, accepted integration steps, tolerance control, or derivative callback counts.",
            "With one output, returns only the `y` trajectory matrix."
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and compatibility",
            paragraphs: &[
                "The integration loop runs on the CPU and invokes the supplied derivative through RunMat's ordinary callback service. Automatic resident input may gather; explicit gpuArray fallback is a gated RunMat extension.",
                "Finite host-memory execution is currently unsupported for ODE solvers: requests with a finite host-memory policy fail before derivative callbacks or solver workspace/history allocation; ordinary unbounded execution is unchanged.",
                "The current single-output form returns the y trajectory rather than MATLAB's solution structure. Event arguments and event outputs are not yet supported. These are interface gaps, not alternate MATLAB-compatible forms.",
                "The current solver returns double time and state arrays, including for single initial conditions. Integer and logical inputs are separately gated RunMat extensions.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "example-1",
        title: "Integrate exponential decay",
        program: "[t, y] = ode45(@(t, y) -y, [0 5], 1)",
        display_output: Some("y(end) ≈ exp(-5)"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(size(y,2)==1); assert(size(y,1)==size(t,1)); assert(abs(y(end)-exp(-5))<5e-3);",
        },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "ode15s",
            target: BuiltinDocumentationLinkTarget::Builtin("ode15s"),
        },
        BuiltinDocumentationLink {
            label: "ode23",
            target: BuiltinDocumentationLinkTarget::Builtin("ode23"),
        },
        BuiltinDocumentationLink {
            label: "Source",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/ode/ode45.rs"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "ODE solver unit tests",
            location: "builtins::math::ode::ode45::tests",
        }],
        notes: &[
            "The legacy sidecar's summary, keywords, behaviors, example, links, GPU and fusion claims were reconciled. Standalone example acceptance and hardware qualification remain separate gates.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

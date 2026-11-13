use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ode23"),
    slug: Some("ode23"),
    summary: "Solve nonstiff ODE systems using adaptive Bogacki-Shampine 3(2) integration.",
    description: "`ode23` integrates nonstiff first-order ODE systems with adaptive low-order Runge-Kutta stepping and returns MATLAB-compatible time/state output arrays.",
    keywords: &["ode23", "ode", "nonstiff", "Bogacki-Shampine", "adaptive step"],
    related: &["ode15s", "ode45"],
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
            "Two-endpoint time spans use Refine=1 by default. A larger Refine uses the Bogacki-Shampine cubic Hermite continuous extension without additional derivative calls.",
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
        title: "Integrate a scalar ODE at requested output times",
        program: "[t, y] = ode23(@(t, y) -2*y, [0 0.25 0.5 1.0], 1)",
        display_output: Some("size(t) = [4, 1], size(y) = [4, 1]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(size(y,1)==4); assert(size(t,1)==4); assert(abs(y(1)-1)<1e-12);",
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
            label: "ode45",
            target: BuiltinDocumentationLinkTarget::Builtin("ode45"),
        },
        BuiltinDocumentationLink {
            label: "Source",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/ode/ode23.rs"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "ODE solver unit tests",
            location: "builtins::math::ode::ode23::tests",
        }],
        notes: &[
            "The legacy sidecar's summary, keywords, behaviors, example, links, GPU and fusion claims were reconciled. Standalone example acceptance and hardware qualification remain separate gates.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

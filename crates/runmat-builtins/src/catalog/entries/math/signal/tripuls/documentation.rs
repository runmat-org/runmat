use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tripuls"),
    slug: Some("tripuls"),
    summary: "Generate sampled triangular pulses.",
    description: "`tripuls(T)` returns samples of a unit-width triangular pulse centered at zero. `tripuls(T, W, S)` uses width `W` and skew `S` in `[-1, 1]`. The documented single and double sample-time classes preserve their floating output class. Typed-integer sample times and controls are available in RunMat mode through checked binary64 boundaries.",
    keywords: &["tripuls", "triangular pulse", "pulse train", "signal processing", "skew"],
    related: &["pulstran", "rectpuls", "gauspuls"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The pulse is evaluated element-wise on real numeric scalars and arrays, preserving the sample-time shape. It is zero outside `[-W/2, W/2]` and reaches one at `S*W/2`.",
                "Width must be positive and finite; skew must be finite and in `[-1, 1]`. Complex and text inputs produce a builtin-scoped error.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "Single sample times produce single output; double sample times produce double output. Typed-integer sample times produce double output in RunMat mode. Typed-integer sample times and controls have separate compatibility gates and require exact binary64 conversion.",
                "The current implementation has no triangular-pulse GPU kernel or fusion path. Resident inputs are gathered for host evaluation, and supported results can return to their provider. Explicit gpuArray input is separately gated in RunMat mode.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Default centered triangle",
            program: "t = -0.5:0.25:0.5;\ny = tripuls(t)",
            display_output: Some("y = [0 0.5 1 0.5 0]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(y),size(t))); assert(isequal(y,[0 0.5 1 0.5 0])); assert(strcmp(class(y),'double'));",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Skewed triangle",
            program: "t = -1:1;\ny = tripuls(t, 2, 1)",
            display_output: Some("y = [0 0.5 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(y),size(t))); assert(isequal(y,[0 0.5 1]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Can `tripuls` be used inside `pulstran`?",
        answer: "Yes. `pulstran(T, D, 'tripuls', W, S)` evaluates triangular pulses centered at each delay.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "pulstran", target: BuiltinDocumentationLinkTarget::Builtin("pulstran") },
        BuiltinDocumentationLink { label: "rectpuls", target: BuiltinDocumentationLinkTarget::Builtin("rectpuls") },
        BuiltinDocumentationLink { label: "gauspuls", target: BuiltinDocumentationLinkTarget::Builtin("gauspuls") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/tripuls.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Triangular pulse admission, shape, class, and provider restoration",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/tripuls.rs"),
        }, BuiltinDocumentationLink {
            label: "Shared typed waveform scalar controls",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/waveform_control.rs"),
        }, BuiltinDocumentationLink {
            label: "Typed, cooperative host waveform mapping",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/waveform_map.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Pulse shape, skew, classes, integer controls, and provider behavior",
            location: "builtins::math::signal::tripuls::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: None,
};

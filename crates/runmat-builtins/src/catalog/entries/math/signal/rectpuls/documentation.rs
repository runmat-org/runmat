use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rectpuls"),
    slug: Some("rectpuls"),
    summary: "Generate sampled rectangular pulses.",
    description: "`rectpuls(T)` returns samples of a unit-width rectangular pulse centered at zero. `rectpuls(T, W)` uses pulse width `W`.",
    keywords: &[
        "rectpuls",
        "rectangular pulse",
        "pulse train",
        "signal processing",
        "elementwise",
    ],
    related: &["pulstran", "tripuls", "gauspuls"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Operates element-wise on real numeric scalars, vectors, matrices, and N-D tensors, preserving the sample-time shape.",
                "The pulse is one on the left-closed, right-open interval `-W/2 <= T < W/2` and zero outside. At the default width, `rectpuls(-0.5)` is one and `rectpuls(0.5)` is zero.",
                "Width must be a positive finite scalar. Complex and text sample times are rejected.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Class and execution",
            paragraphs: &[
                "The result is double, including for single sample times. Typed-integer sample times and widths are independently gated RunMat extensions and must convert exactly at the binary64 waveform boundary.",
                "There is no direct GPU kernel or fusion path. Resident inputs may be downloaded for host evaluation; automatic results can be restored through the exact provider owner. Explicit gpuArray input is separately gated in RunMat mode, and its output must retain explicit residency or fail.",
                "The supported resident restoration is not evidence of a GPU pulse kernel or of a transfer-inclusive placement choice.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Default unit-width rectangular pulse",
            program: "t = -0.5:0.25:0.5;\ny = rectpuls(t)",
            display_output: Some("y = [1 1 1 1 0]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(y),size(t))); assert(isequal(y,[1 1 1 1 0])); assert(strcmp(class(y),'double'));",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Custom pulse width",
            program: "t = -1:0.5:1;\ny = rectpuls(t, 0.5)",
            display_output: Some("y = [0 0 1 0 0]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(y),size(t))); assert(isequal(y,[0 0 1 0 0]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What happens at the exact pulse edge?",
            answer: "The interval is left-closed and right-open: RunMat returns one at `T == -W/2` and zero at `T == W/2`.",
        },
        BuiltinDocumentationFaq {
            question: "Can `rectpuls` be used inside `pulstran`?",
            answer: "Yes. `pulstran(T, D, 'rectpuls', W)` evaluates rectangular pulses centered at each delay.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "pulstran",
            target: BuiltinDocumentationLinkTarget::Builtin("pulstran"),
        },
        BuiltinDocumentationLink {
            label: "tripuls",
            target: BuiltinDocumentationLinkTarget::Builtin("tripuls"),
        },
        BuiltinDocumentationLink {
            label: "gauspuls",
            target: BuiltinDocumentationLinkTarget::Builtin("gauspuls"),
        },
        BuiltinDocumentationLink {
            label: "Source",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/rectpuls.rs"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Rectangular pulse admission, boundaries, and restoration",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/rectpuls.rs"),
            },
            BuiltinDocumentationLink {
                label: "Shared typed waveform scalar controls",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/waveform_control.rs"),
            },
            BuiltinDocumentationLink {
                label: "Typed, cooperative host waveform mapping",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/waveform_map.rs"),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Pulse edges, width, typed input and provider behavior",
                location: "builtins::math::signal::rectpuls::tests",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};

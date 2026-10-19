use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pulstran"),
    slug: Some("pulstran"),
    summary: "Generate pulse trains from named pulse functions, function handles, or sampled prototype pulses.",
    description: "`pulstran(T, D, FUN, P1, ...)` evaluates a pulse function at `T - D(k)` for each delay and sums the results. `D` may be a delay vector or an N-by-2 delay/amplitude matrix. `pulstran(T, D, P, FS)` uses a sampled prototype pulse `P` with sample rate `FS` and linear interpolation.",
    keywords: &["pulstran", "pulse train", "rectpuls", "tripuls", "gauspuls", "signal processing"],
    related: &["rectpuls", "tripuls", "gauspuls", "plot"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The public contract does not document typed-integer waveform inputs. RunMat mode admits integer `T`, delay/gain data, sampled prototypes, sample rates, and built-in pulse parameters through independent compatibility gates and rejects any value that cannot be represented exactly at the binary64 waveform boundary.",
            "The output shape matches `T`.",
            "Character, string, and function-handle pulse names are accepted, including optional leading `@`. Built-in fast paths cover `rectpuls`, `tripuls`, and `gauspuls`; other function handles are invoked through the normal `feval` path.",
            "A delay vector uses unit amplitude for every pulse.",
            "An N-by-2 delay matrix uses column one as delay and column two as amplitude.",
            "Sampled prototype pulses are linearly interpolated at `FS`; samples outside the prototype support contribute zero.",
            "Complex and text-valued numeric inputs are rejected with builtin-scoped errors; function-handle callback payloads remain ordinary exact values rather than being reclassified as `pulstran` controls. Typed-integer samples returned by a callback are admitted as waveform results only when every sample is exactly representable as binary64, independent of the input extension gates.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &[
            "Automatic resident waveform inputs may gather for host evaluation and restore double output through the exact owner. Every resident computation operand remains protected from physical output aliasing through restoration; any explicit gpuArray computation input is independently gated and requires explicit output residency. Callback payloads are passed through, not reclassified as computation controls.",
            "The callback form invokes the ordinary function-handle path once per delay with shifted times; callback effects, reentry, suspension, and errors remain observable. The result is double with the shape of T. This is a fusion boundary, not a resident pulse kernel.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Generate a rectangular pulse train", program: "t = -1:0.5:1;\nd = [-0.5 0.5];\ny = pulstran(t, d, 'rectpuls', 0.25)", display_output: Some("y = [0 1 0 1 0]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(y,[0 1 0 1 0])); assert(strcmp(class(y),'double'));" } },
        BuiltinExample { id: "example-2", title: "Use delay amplitudes", program: "t = [0 1];\nd = [0 2; 1 3];\ny = pulstran(t, d, 'rectpuls', 0.25)", display_output: Some("y = [2 3]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(y,[2 3])); assert(strcmp(class(y),'double'));" } },
        BuiltinExample { id: "example-3", title: "Use a sampled prototype", program: "t = 0:0.5:2;\nd = [0 1];\np = [0 1 0];\ny = pulstran(t, d, p, 2)", display_output: Some("y = [0 1 0 1 0]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(y,[0 1 0 1 0])); assert(strcmp(class(y),'double'));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which pulse functions are recognized directly?", answer: "`rectpuls`, `tripuls`, and `gauspuls` have direct built-in paths. Other function handles are dispatched through RunMat's regular function-handle mechanism." },
        BuiltinDocumentationFaq { question: "Does `pulstran` preserve the orientation of `T`?", answer: "Yes. The output tensor uses the same shape as `T`, including row-vector and column-vector orientation." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "rectpuls", target: BuiltinDocumentationLinkTarget::Builtin("rectpuls") },
        BuiltinDocumentationLink { label: "tripuls", target: BuiltinDocumentationLinkTarget::Builtin("tripuls") },
        BuiltinDocumentationLink { label: "gauspuls", target: BuiltinDocumentationLinkTarget::Builtin("gauspuls") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/pulstran.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime admission and controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/pulstran.rs") },
            BuiltinDocumentationLink { label: "Waveform and callback evaluation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/pulstran/evaluation.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Input, output and failure semantics", location: "builtins::math::signal::pulstran::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

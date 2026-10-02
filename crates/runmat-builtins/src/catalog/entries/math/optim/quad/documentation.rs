use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("quad"), slug: Some("quad"),
    summary: "Approximate finite scalar definite integrals with the legacy quad interface.",
    description: "`q = quad(fun, a, b)` approximates a scalar definite integral using adaptive Simpson quadrature. Optional tolerance, trace, forwarded callback arguments, and `[q, fcnt]` are supported.",
    keywords: &["quad", "numerical integration", "adaptive simpson", "quadrature", "function handle"],
    related: &["integral", "trapz", "cumtrapz"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The callback may be a named handle, anonymous handle, or function-handle string. It is evaluated at scalar sample points and must return a finite real scalar. Complex-valued and array-valued callbacks are not admitted.",
            "The positive absolute tolerance defaults to `1e-6`; an empty tolerance selects that default. A nonzero `trace` prints subdivision rows as `[fcnEvals, a, b-a, Q]`; empty trace disables it.",
            "Arguments after `trace` are passed unchanged to the callback as `fun(x, p1, p2, ...)`. Reversed finite bounds negate the result. Equal bounds return zero with a function count of zero. More than two requested outputs are rejected.",
            "Typed integer bounds, tolerance, and callback results are separately gated RunMat extensions and cross into binary64 only when exactly representable. Integer trace is an exact structural zero/nonzero control; native integer forwarded arguments retain their class and storage.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The adaptive loop runs on the host. Automatic residency can be gathered; explicit `gpuArray` host fallback is a gated RunMat extension. A callback may call GPU-aware builtins, but the quadrature loop has no GPU kernel or fused callback execution.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Integrate sine over one half-period", program: "q = quad(@sin, 0, pi)", display_output: Some("q =\n    2.0000"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(q-2)<1e-5);" } },
        BuiltinExample { id: "example-2", title: "Return the function evaluation count", program: "[q, fcnt] = quad(@(x) x.^2, 0, 1)", display_output: Some("q =\n    0.3333"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(q-1/3)<1e-5); assert(fcnt>=3);" } },
        BuiltinExample { id: "example-3", title: "Forward an extra argument to the integrand", program: "q = quad(@(x,a) a.*x, 0, 2, [], [], 3)", display_output: Some("q =\n    6.0000"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(q-6)<1e-5);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Should new code use quad or integral?", answer: "Use `integral` for new code when possible. `quad` retains the legacy call interface." },
        BuiltinDocumentationFaq { question: "Does quad support extra parameters?", answer: "Yes. Values after `trace` are forwarded unchanged after the scalar sample point." },
        BuiltinDocumentationFaq { question: "Does quad run on the GPU?", answer: "The adaptive solver runs on the host. Its callback may call GPU-aware builtins." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "integral", target: BuiltinDocumentationLinkTarget::Builtin("integral") },
        BuiltinDocumentationLink { label: "trapz", target: BuiltinDocumentationLinkTarget::Builtin("trapz") },
        BuiltinDocumentationLink { label: "cumtrapz", target: BuiltinDocumentationLinkTarget::Builtin("cumtrapz") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/quad.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Quadrature runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/quad.rs") },
            BuiltinDocumentationLink { label: "Callback invocation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common.rs") },
            BuiltinDocumentationLink { label: "Adaptive Simpson traversal", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/quadrature/adaptive.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar quadrature and forwarded parameters", location: "builtins::math::optim::quad::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

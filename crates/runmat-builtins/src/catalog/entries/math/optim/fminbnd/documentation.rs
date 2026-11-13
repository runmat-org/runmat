use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fminbnd"),
    slug: Some("fminbnd"),
    summary: "Find bounded scalar minima with Brent's method in MATLAB and RunMat." ,
    description: "`x = fminbnd(fun, x1, x2)` searches the closed interval `[x1, x2]` for a local minimum of scalar function `fun` using Brent's method. The documented compatibility contract uses binary64 bounds. Optional outputs return the minimum value, exit flag, and diagnostics.",
    keywords: &["fminbnd", "bounded minimization", "brent", "golden section", "parabolic interpolation", "optimization"],
    related: &["fminunc", "fzero", "fsolve", "optimset"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The function handle must return a finite real scalar for every sampled point in the interval.",
            "Strict compatibility mode rejects typed-integer or logical bounds and objective values. RunMat extension mode accepts those values when every integer converts exactly to binary64.",
            "Typed-integer option controls are a separately gated RunMat extension. Structural counts such as `MaxIter` and `MaxFunEvals` are validated as exact nonnegative integers rather than rounded through floating point.",
            "If `x1 > x2`, the bounds are inconsistent and `exitflag = -2`.",
            "If `x1 == x2`, that point is returned with `exitflag = 1`.",
            "Options are supplied as a struct, usually built with `optimset`. `TolX`, `MaxIter`, `MaxFunEvals`, and `Display` are accepted.",
            "`Display` may be `'off'`, `'iter'`, `'notify'` (default; only prints when the solver fails to converge), or `'final'`.",
            "`exitflag` is `1` on convergence within `TolX`, `0` when the iteration or evaluation budget is exhausted.",
            "The fourth output is a struct with `iterations`, `funcCount`, `algorithm`, and `message` fields.",
            "A call with no assigned output still runs the solver and its objective callback; it publishes no result value.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The solver runs on the host and has no provider-native solver kernel or tall-array execution. Authenticated automatic placement can be gathered transparently under either language compatibility mode. Explicit GPU operands and callback results require the RunMat resident-fallback extension; strict MATLAB mode rejects them before provider access. Device operands with unproven placement require an explicit gather. Compatibility, cancellation and provider authority come from the original invocation.",
            "Fusion: not elementwise; not a reduction. The callback solver executes on the host.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Minimize a quadratic", program: "x = fminbnd(@(x) (x-2).^2, 0, 5)", display_output: Some("x =\n    2.0000"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-2)<1e-3);" } },
        BuiltinExample { id: "example-2", title: "Capture the function value too", program: "[x, fval] = fminbnd(@(x) (x-3).^2 + 1, 0, 5)", display_output: Some("x =\n    3.0000\nfval =\n    1.0000"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-3)<1e-3); assert(abs(fval-1)<1e-6);" } },
        BuiltinExample { id: "example-3", title: "Tighten tolerances", program: "opts = optimset('TolX', 1e-10, 'Display', 'off');\n[x, fval, exitflag] = fminbnd(@cos, 0, pi, opts)", display_output: Some("x =\n    3.1416\nfval =\n   -1.0000\nexitflag =\n     1"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-pi)<1e-3); assert(fval < -0.999); assert(exitflag==1);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does fminbnd support unbounded minimization?", answer: "No. `fminbnd` requires finite scalar bounds. Use `fminunc` or `fminsearch` for unconstrained problems." },
        BuiltinDocumentationFaq { question: "Can I retrieve the iteration history?", answer: "Yes — request the four-output form `[x, fval, exitflag, output] = fminbnd(...)` to receive a struct with `iterations`, `funcCount`, `algorithm`, and `message`." },
        BuiltinDocumentationFaq { question: "Does fminbnd guarantee a global minimum?", answer: "No. Brent's method converges to a local minimum on `[x1, x2]`. For multi-modal functions, narrow the interval to bracket the desired minimum." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "fminunc", target: BuiltinDocumentationLinkTarget::Builtin("fminunc") },
        BuiltinDocumentationLink { label: "fzero", target: BuiltinDocumentationLinkTarget::Builtin("fzero") },
        BuiltinDocumentationLink { label: "fsolve", target: BuiltinDocumentationLinkTarget::Builtin("fsolve") },
        BuiltinDocumentationLink { label: "optimset", target: BuiltinDocumentationLinkTarget::Builtin("optimset") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/fminbnd.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[
        BuiltinDocumentationLink { label: "Runtime and admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/fminbnd.rs") },
        BuiltinDocumentationLink { label: "Shared Brent solver", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/brent.rs") },
        BuiltinDocumentationLink { label: "Callback and option services", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common.rs") },
        BuiltinDocumentationLink { label: "Typed option fields", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common/scalar_options.rs") },
        BuiltinDocumentationLink { label: "Invocation cooperation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common/scalar_cooperation.rs") },
    ], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Solver behavior", location: "builtins::math::optim::fminbnd::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Function handle execution", location: "runmat-vm/tests/closures.rs::fminbnd_accepts_anonymous_function" },
    ], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fzero"),
    slug: Some("fzero"),
    summary: "Find scalar function zeros with bracketed root-finding in MATLAB and RunMat.",
    description: "`x = fzero(fun, x0)` searches near a binary32 or binary64 scalar initial point `x0` for a sign-changing bracket, then refines the root with a Brent-style method. `x = fzero(fun, [a b])` uses the supplied bracket directly. Optional outputs follow MATLAB's standard contract: `[x, fval]`, `[x, fval, exitflag]`, and `[x, fval, exitflag, output]`.",
    keywords: &["fzero", "root finding", "zero", "brent", "optimization"],
    related: &["fsolve", "optimset", "roots"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The function handle must return a finite real scalar. A binary32 initial point or interval is threaded through callback arguments and numeric outputs; when `TolX` is not explicit, either a binary32 initial point or interval or a binary32 initial function value selects `eps('single')` instead of `eps('double')`.",
            "Strict compatibility mode rejects typed-integer or logical initial points, intervals, and function values. RunMat extension mode accepts those values when every integer converts exactly to binary64.",
            "Typed-integer option controls are a separately gated RunMat extension. Structural counts such as `MaxIter` and `MaxFunEvals` are validated as exact nonnegative integers rather than rounded through floating point.",
            "Bracket endpoints must have opposite signs unless one endpoint is already a root.",
            "Options are supplied as a struct, usually created with `optimset`. `TolX`, `MaxIter`, `MaxFunEvals`, and `Display` are accepted.",
            "`fval` is the function value at the returned `x`; `exitflag` is 1 for convergence and 0 when the iteration/function-evaluation budget is exhausted.",
            "The four-output `output` struct contains `iterations`, `funcCount`, `algorithm`, and `message` fields.",
            "Requests for more than four outputs are rejected.",
            "A call with no assigned output still runs the solver and its callback; it publishes no result value.",
            "Anonymous functions, named function handles, and function-handle strings work through RunMat's existing `feval` dispatch path.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The solver runs on the host and has no provider-native solver kernel or tall-array execution. Authenticated automatic placement can be gathered transparently under either language compatibility mode. Explicit GPU operands and callback results require the RunMat resident-fallback extension; strict MATLAB mode rejects them before provider access. Device operands with unproven placement require an explicit gather. Compatibility, cancellation and provider authority come from the original invocation.",
            "The solver invokes user callbacks and terminates fusion; it does not provide a native GPU kernel.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Find the fixed point of cosine", program: "f = @(x) cos(x) - x;\nx = fzero(f, 0.5)", display_output: Some("x =\n    0.7391"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-0.7390851332151607)<1e-3);" } },
        BuiltinExample { id: "example-2", title: "Use an explicit bracket", program: "x = fzero(@sin, [3 4])", display_output: Some("x =\n    3.1416"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-pi)<1e-3);" } },
        BuiltinExample { id: "example-3", title: "Return the root and function value", program: "f = @(x) x.^2 - 4;\n[x, fval] = fzero(f, [0 3])", display_output: Some("x =\n    2.0000\nfval =\n     0"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-2)<1e-6); assert(abs(fval)<1e-6);" } },
        BuiltinExample { id: "example-4", title: "Set a tighter tolerance", program: "opts = optimset('TolX', 1e-10, 'Display', 'off');\nx = fzero(@sin, [3 4], opts)", display_output: Some("x =\n    3.1416"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(x-pi)<1e-6);" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "fsolve", target: BuiltinDocumentationLinkTarget::Builtin("fsolve") },
        BuiltinDocumentationLink { label: "optimset", target: BuiltinDocumentationLinkTarget::Builtin("optimset") },
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/fzero.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[
        BuiltinDocumentationLink { label: "Runtime and admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/fzero.rs") },
        BuiltinDocumentationLink { label: "Shared Brent solver", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/brent.rs") },
        BuiltinDocumentationLink { label: "Callback and option services", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common.rs") },
        BuiltinDocumentationLink { label: "Typed option fields", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common/scalar_options.rs") },
        BuiltinDocumentationLink { label: "Invocation cooperation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common/scalar_cooperation.rs") },
    ], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Solver behavior", location: "builtins::math::optim::fzero::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Function handle execution", location: "runmat-vm/tests/closures.rs::fzero_returns_optional_function_value" },
    ], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

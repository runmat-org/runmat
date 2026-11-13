use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("limit"),
    slug: Some("limit"),
    summary: "Evaluate scalar symbolic limits by substitution and L'Hopital reduction.",
    description: "`limit` evaluates scalar symbolic expressions at a finite numeric point. It first performs direct substitution and simplification, then applies repeated L'Hopital reduction for removable `0/0` quotients.",
    keywords: &["limit", "symbolic", "calculus", "lhopital"],
    related: &["sym", "syms", "int", "vpa"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Supports `limit(expr)`, `limit(expr, point)`, `limit(expr, var)`, `limit(expr, var, point)`, and an optional direction string. MATLAB-compatible calls require a symbolic expression; RunMat mode independently admits scalar numeric expressions.",
                "All eight integer classes are accepted for a numeric limit point and, in RunMat mode, a numeric expression. Native storage is read directly and values that cannot enter the current binary64 symbolic core exactly are rejected before substitution.",
                "Infers the limit variable when the expression contains exactly one symbolic variable.",
                "Simplifies symbolic arithmetic involving `+`, `-`, `*`, `/`, powers with numeric exponents, and the functions `sin`, `cos`, `tan`, `exp`, `log`, and `sqrt`.",
                "Successful results are scalar symbolic values, including symbolic constants when the limit resolves to a finite value or a supported one-sided infinity.",
                "One-sided directions are accepted as `left`, `right`, `-`, or `+` for simple pole detection.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Supported scalar forms and limits",
            paragraphs: &[
                "Accepts expression alone, expression plus variable or point, expression/variable/point, or that three-argument form plus direction. Inferring a variable requires exactly one symbolic variable; explicit variable text must be a valid identifier and explicit symbolic variables must be variable nodes.",
                "Directions are trimmed and case-insensitive: left/- or right/+; empty text, both, two-sided and twosided select the default two-sided case. Symbolic expressions and numbers are not direction strings.",
                "Points must be finite real scalar constants. Host scalar logical values and one-element real numeric tensors are accepted, but complex, sparse, empty/nonscalar, symbolic-array and logical-array containers reject. Native integer expression/point values that cannot convert exactly to binary64 reject before substitution.",
                "Every successful result is a host symbolic scalar. One-sided simple poles can produce symbolic infinities; unresolved cases raise RunMat:limit:Unsupported rather than claiming full toolbox behavior. Zero requested outputs evaluates and discards the result; more than one fails before processing.",
                "RunMat uses at most eight L'Hopital reductions and a limited simple-pole sampling heuristic (offset1e-7, zero threshold1e-12). General assumptions, infinite limit points, symbolic arrays and full one-sided/asymptotic calculus are not implemented.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Placement and effects",
            paragraphs: &["Symbolic limits execute on the host and do not participate in GPU fusion.", "GPU and distributed operands, including secondary controls, are not supported. These functions do not upload or gather values and have no device kernel or GPU fusion recipe.", "Constructing and transforming symbolic expression trees currently requires unrestricted host allocation. Finite Managed calls reject before unowned symbolic allocation; complete finite expression ownership is not yet implemented.", "This portable host operation is pure apart from errors. It does not use callbacks, workspace mutation, filesystem, clock or random services."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Classic sinc limit",
            program: "syms x\nf = limit(sin(x)/x, x, 0)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(f,'sym')); assert(abs(double(f)-1)<1e-12);",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Finite-difference derivative limit",
            program: "syms x h\nf = limit((cos(x+h) - cos(x))/h, h, 0)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(f,'sym')); assert(strcmp(char(f),'-sin(x)'));",
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Fractional symbolic power limit",
            program: "syms x\nf = limit((cos(x)^(1/3) - 1) / x^2, x, 0)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(f,'sym')); assert(abs(double(f)+1/6)<1e-12);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does `limit` implement the full Symbolic Math Toolbox?",
        answer: "No. It implements RunMat's scalar symbolic expression core for common calculus expressions used in MATLAB scripts.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "sym",
            target: BuiltinDocumentationLinkTarget::Builtin("sym"),
        },
        BuiltinDocumentationLink {
            label: "syms",
            target: BuiltinDocumentationLinkTarget::Builtin("syms"),
        },
        BuiltinDocumentationLink {
            label: "int",
            target: BuiltinDocumentationLinkTarget::Builtin("int"),
        },
        BuiltinDocumentationLink {
            label: "vpa",
            target: BuiltinDocumentationLinkTarget::Builtin("vpa"),
        },
        BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/math/symbolic/limit.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/limit.rs"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/symbolic/sym.limit.html"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime implementation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/limit.rs"),
            },
            BuiltinDocumentationLink {
                label: "Scalar text and numeric conversion",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/support.rs"),
            },
            BuiltinDocumentationLink {
                label: "Invocation allocation boundary",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/invocation.rs"),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::symbolic::limit::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "runmat_vm::bytecode::compile::tests::compile_interprets_symbolic_limit_workflow",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "fractional_power",
                location: "runmat_vm::bytecode::compile::tests::compile_interprets_symbolic_fractional_power_limit",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Scalar form inference",
                location: "crates/runmat-builtins/src/catalog/entries/math/symbolic/inference/tests.rs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

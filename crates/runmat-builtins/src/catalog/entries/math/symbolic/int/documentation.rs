use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("int"),
    slug: Some("int"),
    summary: "Evaluate scalar symbolic indefinite and definite integrals.",
    description: "`int(expr)` returns an indefinite integral with respect to the first symbolic variable in `expr`. `int(expr, var)` chooses the variable explicitly. `int(expr, a, b)` and `int(expr, var, a, b)` evaluate definite integrals by substituting the bounds into the antiderivative.",
    keywords: &["int", "symbolic", "integration", "calculus", "definite integral"],
    related: &["sym", "syms", "diff", "limit", "vpa"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["RunMat prefers x when present, then chooses remaining valid symbolic names in deterministic lexical order; a constant expression defaults to x. This is the current scalar default-variable rule.", "Supported closed forms include constants, variables, sums, differences, scalar multiples, powers of the integration variable, reciprocal `1/x`, `sin`, `cos`, `exp`, `log(x)`, and `sqrt(x)`.", "Definite integrals simplify numeric trigonometric and exponential bounds when the antiderivative can be evaluated.", "Unsupported symbolic forms return a formal symbolic `int(...)` expression instead of failing, preserving downstream symbolic workflows.", "IgnoreAnalyticConstraints, PrincipalValue, IgnoreSpecialCases and Hold are recognized MATLAB option names, but RunMat rejects them with RunMat:int:InvalidOption. They are not silently ignored."],
        },
        BuiltinDocumentationSection {
            heading: "Supported scalar forms and limits",
            paragraphs: &[
                "Accepts one expression plus zero, one, two or three arguments according to the four listed forms. Expressions and bounds may be symbolic scalars or host real numeric/scalar logical values, including one-element N-D numeric tensors. Explicit variables must be symbolic variable nodes or valid identifier text, never numeric scalars.",
                "The output is always one host symbolic scalar, including a formal unevaluated integral when no elementary primitive is implemented. Zero requested outputs evaluates and discards it; more than one output fails before processing.",
                "All integer classes are recognized, but this integrator currently requires numeric integer expressions and bounds to be exactly representable in binary64; wider numeric integers reject before symbolic conversion. This restriction does not turn int into an integer datatype conversion.",
                "Complex values, symbolic-array and logical-array containers (even one-element), sparse, empty/nonscalar inputs, vector bound syntax and resident/distributed operands are unsupported. Text is accepted only in the variable and rejected-option roles, not as expression or bound input.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Placement and effects",
            paragraphs: &["Symbolic integration works on host algebra values and is not uploaded to GPU providers.", "GPU and distributed operands, including secondary controls, are not supported. These functions do not upload or gather values and have no device kernel or GPU fusion recipe.", "Constructing and transforming symbolic expression trees currently requires unrestricted host allocation. Finite Managed calls reject before unowned symbolic allocation; complete finite expression ownership is not yet implemented.", "This portable host operation is pure apart from errors. It does not use callbacks, workspace mutation, filesystem, clock or random services."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Integrate a polynomial",
            program: "syms x\nF = int(x^2)",
            display_output: Some("F = x^3/3"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(F,'sym')); assert(strcmp(char(F),'x^3/3'));",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Evaluate a definite integral",
            program: "syms x\nA = int(sin(x), 0, pi)",
            display_output: Some("A = 2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(A,'sym')); assert(abs(double(A)-2)<1e-12);",
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Choose the integration variable explicitly",
            program: "syms x y\nF = int(y, x)",
            display_output: Some("F = x*y"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(F,'sym')); assert(strcmp(char(F),'x*y'));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `int` run on the GPU?",
            answer: "No. Symbolic expressions are host algebra values. Numeric tensors should use numerical integration routines when available.",
        },
        BuiltinDocumentationFaq {
            question: "What happens when RunMat cannot find a closed form?",
            answer: "RunMat returns a formal scalar symbolic `int(...)` expression so code can continue to manipulate or display the unevaluated integral.",
        },
    ],
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
            label: "diff",
            target: BuiltinDocumentationLinkTarget::Builtin("diff"),
        },
        BuiltinDocumentationLink {
            label: "limit",
            target: BuiltinDocumentationLinkTarget::Builtin("limit"),
        },
        BuiltinDocumentationLink {
            label: "vpa",
            target: BuiltinDocumentationLinkTarget::Builtin("vpa"),
        },
        BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/math/symbolic/int.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/int.rs"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/symbolic/sym.int.html"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime implementation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/int.rs"),
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
                location: "builtins::math::symbolic::int::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "runmat-vm tests::symbolic_int_source_workflow",
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

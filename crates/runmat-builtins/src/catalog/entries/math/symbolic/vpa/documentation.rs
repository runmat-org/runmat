use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("vpa"),
    slug: Some("vpa"),
    summary: "Evaluate scalar symbolic or numeric input as a symbolic decimal.",
    description: "`vpa(x)` converts a scalar symbolic or numeric input to a scalar symbolic decimal literal using the current `digits` setting. `vpa(x, d)` uses `d` significant decimal digits. Exact symbolic rationals are expanded exactly; ordinary numeric inputs are rendered from the numeric value supplied by the script.",
    keywords: &["vpa", "symbolic", "variable precision", "digits", "decimal"],
    related: &["digits", "sym", "syms", "int"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["The default precision is controlled by `digits` and starts at 32 significant digits.", "Exact rational symbolic inputs such as `sym('1/3')` are expanded and rounded with the requested significant digits.", "Numeric scalar inputs are converted to symbolic decimal literals from their existing numeric value at the requested display precision.", "Scalar values from every integer class are lifted directly into exact arbitrary-precision integer storage, including `int64` and `uint64` extrema.", "Integer precision controls are decoded exactly; noninteger floating controls are rounded to the nearest integer before the supported range is checked.", "Nonconstant symbolic variables remain symbolic.", "RunMat currently represents scalar symbolic expressions; array-valued symbolic containers and the documented `BoostPrecision` option are not implemented."],
        },
        BuiltinDocumentationSection {
            heading: "Supported scalar forms and limits",
            paragraphs: &[
                "Inputs are host symbolic scalars, real numeric/scalar logical values or scalar numeric/rational text. One-element N-D real numeric tensors are accepted; single values enter the symbolic binary64 constant domain after their original single rounding. Complex inputs and symbolic/logical array containers, even one-element, reject.",
                "Explicit precision is a real host numeric scalar rounded to the nearest integer in1..=100000000; native integer precision is decoded exactly. Scalar logical true selects one digit, false is out of range. Text precision, empty/nonscalar, complex and resident controls are unsupported.",
                "Scalar text can be a finite number or an integer numerator/denominator rational with nonzero denominator. Bare variables require symbolic input, not arbitrary text. A character argument must be a valid row without non-singleton character pages; one-element string arrays are accepted.",
                "Exact rationals are rendered with significant-digit rounding, including integer digits, carry across magnitudes and very small finite ratios. The symbolic decimal retains its display text plus a binary64 numeric approximation; subsequent numeric conversions are not arbitrary-precision arithmetic.",
                "Numeric inputs are rendered from their existing value: pi supplied as a numeric double does not acquire additional mathematical digits. Constant transcendental expressions use existing binary64 evaluation. This is not full arbitrary-precision transcendental computation and does not implement BoostPrecision.",
                "Omitting precision reads the current thread-local digits setting; explicit precision does not read or change it. One host symbolic scalar is returned. Zero requested outputs evaluates and discards it; requesting more than one output fails before processing.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Placement and effects",
            paragraphs: &["Variable-precision symbolic values are host algebra values and are not uploaded to GPU providers.", "GPU and distributed operands, including secondary controls, are not supported. These functions do not upload or gather values and have no device kernel or GPU fusion recipe.", "Constructing and transforming symbolic expression trees currently requires unrestricted host allocation. Finite Managed calls reject before unowned symbolic allocation; complete finite expression ownership is not yet implemented.", "No callbacks, workspace mutation, filesystem, clock or random services are used. Only the one-argument form reads the thread-local symbolic precision setting."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Render numeric pi at 50 digits",
            program: "vpa(pi, 50)",
            display_output: Some("3.1415926535897931159979634685441851615905761718750"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "rendered=vpa(pi,50); assert(isa(rendered,'sym')); assert(double(rendered)==pi); assert(strcmp(char(rendered),'3.1415926535897931159979634685441851615905761718750'));",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Evaluate an exact rational",
            program: "vpa(sym('1/3'), 20)",
            display_output: Some("0.33333333333333333333"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "rendered=vpa(sym('1/3'),20); assert(isa(rendered,'sym')); assert(strcmp(char(rendered),'0.33333333333333333333'));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does `vpa` return a double?",
        answer: "No. It returns a scalar symbolic value so formatting and downstream symbolic operations can preserve the decimal literal.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "digits",
            target: BuiltinDocumentationLinkTarget::Builtin("digits"),
        },
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
            label: "`crates/runmat-runtime/src/builtins/math/symbolic/vpa.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/vpa.rs"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/symbolic/sym.vpa.html"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime implementation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/vpa.rs"),
            },
            BuiltinDocumentationLink {
                label: "Scalar text and numeric conversion",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/support.rs"),
            },
            BuiltinDocumentationLink {
                label: "Invocation allocation boundary",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/invocation.rs"),
            },
            BuiltinDocumentationLink {
                label: "Significant decimal rendering",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/vpa/decimal.rs"),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::symbolic::vpa::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "runmat-vm tests::symbolic_vpa_source_workflow",
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

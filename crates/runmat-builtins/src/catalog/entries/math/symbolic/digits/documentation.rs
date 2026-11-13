use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("digits"),
    slug: Some("digits"),
    summary: "Get or set the default variable-precision digit count.",
    description: "`digits` returns the current default precision for `vpa`. `digits(d)` rounds a finite numeric scalar to the nearest integer, sets that precision, and returns the previous value as a double scalar. RunMat mode additionally supports `digits('default')` as an explicitly gated reset extension.",
    keywords: &["digits", "vpa", "symbolic", "precision"],
    related: &["vpa", "sym"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["The default value is 32 significant digits.", "The public precision range is 2 through 100000000 significant digits, inclusive, and noninteger floating controls round to the nearest integer.", "Host controls can use single, double, or any of the eight built-in integer classes; logical and resident controls reject.", "Numeric text and the `'default'` reset are named RunMat-only extensions and reject in MATLAB-compatible mode.", "The setting affects later `vpa(x)` calls that omit an explicit precision."],
        },
        BuiltinDocumentationSection {
            heading: "Supported scalar forms and limits",
            paragraphs: &["Precision is thread-local runtime state, not ordinary floating-point precision or a separate setting per workspace. A query reads that state; setting a new value returns the old value and changes subsequent vpa calls on the same runtime thread. Failed validation and excess output requests leave it unchanged.", "Zero requested outputs still performs a valid update. More than one output is an error before mutation. The result is always a real double scalar.", "Real numeric controls, including one-element N-D single/double/integer tensors, are decoded directly without expression-tree allocation. Empty, nonscalar, complex, logical-array, symbolic and resident controls reject. Scalar logical controls reject as well.", "Numeric controls remain available with finite Managed allocation policy. Numeric-text and default-reset forms currently require unrestricted host allocation because their text conversion is not yet a finite-owned path."],
        },
        BuiltinDocumentationSection {
            heading: "Placement and effects",
            paragraphs: &["The precision setting is host runtime state for symbolic variable-precision formatting.", "GPU and distributed operands, including secondary controls, are not supported. These functions do not upload or gather values and have no device kernel or GPU fusion recipe.", "No workspace variables, callbacks, filesystem, clock or random services are used. Only the thread-local symbolic precision setting is read or changed."],
        },
    ],
    examples: &[BuiltinExample {
        id: "example-1",
        title: "Set and reset precision",
        program: "old = digits(50)\nx = vpa(sym('1/3'))\ndigits('default')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(old,'double')); assert(isa(x,'sym')); assert(strcmp(char(x),'0.33333333333333333333333333333333333333333333333333')); assert(digits()==32);",
        },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does `digits` change ordinary double precision arithmetic?",
        answer: "No. It only controls the default significant-digit count used by symbolic variable-precision operations such as `vpa`.",
    }],
    links: &[
        BuiltinDocumentationLink {
            label: "vpa",
            target: BuiltinDocumentationLinkTarget::Builtin("vpa"),
        },
        BuiltinDocumentationLink {
            label: "sym",
            target: BuiltinDocumentationLinkTarget::Builtin("sym"),
        },
        BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/math/symbolic/digits.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/digits.rs"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/symbolic/sym.digits.html"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime implementation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/symbolic/digits.rs"),
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
                location: "builtins::math::symbolic::digits::tests",
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

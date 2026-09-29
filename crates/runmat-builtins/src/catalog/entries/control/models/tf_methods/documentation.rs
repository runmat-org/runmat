use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime operator bindings",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/tf/methods.rs"),
};
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "SISO model arithmetic", paragraphs: &[
        "These identities are tf class operator methods, reached through ordinary MATLAB-syntax operators on tf objects. They are not standalone top-level functions. A binary operation accepts another SISO tf model or a scalar gain; both model operands must have matching sample times and polynomial variables. Elementwise and matrix spellings share the same SISO operation in this implementation.",
        "Results are new host tf values. Polynomial coefficient arrays remain double or complex-double, while sample time and variable metadata come from the input model. Nonzero input/output delay arithmetic, MIMO models, and incompatible sample times or variables are rejected.",
    ] },
    BuiltinDocumentationSection { heading: "Precision and execution", paragraphs: &[
        "Typed integer scalar gains are a RunMat-mode extension. They convert to double for polynomial arithmetic, so wide integer gains can round; this differs from checked integer coefficients stored in a tf model. Typed-integer or logical coefficients and typed-integer Ts assigned after construction are gated again when a method admits the model, with exact conversion for integer model properties. Integer-valued power exponents are decoded exactly as signed 64-bit counts; zero returns a unity model and negative counts use a reciprocal. Coefficient arithmetic is host-side; resident scalar gains and model properties may gather through their owning provider. The method call can suspend and terminates numeric fusion.",
    ] },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Are these top-level functions?",
        answer:
            "No. They are tf operator methods reached by MATLAB-syntax arithmetic on a tf model.",
    },
    BuiltinDocumentationFaq {
        question: "Are MIMO transfer functions supported?",
        answer: "No. These methods currently operate on SISO transfer functions and scalar gains.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "tf",
        target: BuiltinDocumentationLinkTarget::Builtin("tf"),
    },
    SOURCE,
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[SOURCE],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "SISO arithmetic, exact exponent, and provider admission", location: "crates/runmat-runtime/src/builtins/control/tf/methods.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM control operator workflow", location: "crates/runmat-vm/tests/control.rs" },
    ],
    notes: &["Native, browser, provider and example replay remain integrated gates; catalog publication is method-only."],
};

macro_rules! method_documentation {
    ($name:ident, $title:literal, $summary:literal, $program:literal, $assertion:literal) => {
        pub(super) const $name: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($title),
            slug: Some($title),
            summary: $summary,
            description: $summary,
            keywords: &[
                "tf",
                "transfer function",
                "SISO",
                "control system",
                "operator",
            ],
            related: &["tf"],
            sections: SECTIONS,
            examples: &[BuiltinExample {
                id: "operation",
                title: "SISO transfer-function operator",
                program: $program,
                display_output: None,
                compatibility: BuiltinExampleCompatibility::Matlab,
                harness: BuiltinExampleHarness::Portable,
                fixture: BuiltinExampleFixture::None,
                requirements: BuiltinExampleRequirements::NONE,
                verification: BuiltinExampleVerification::Assertions { source: $assertion },
            }],
            example_exemption: None,
            faqs: FAQS,
            links: LINKS,
            media: &[],
            evidence: EVIDENCE,
            introduced: None,
            status: Some(BuiltinDocumentationStatus::Partial),
        };
    };
}

method_documentation!(PLUS, "tf.plus", "Add two SISO tf models or a scalar gain to a tf model.", "g = tf(2,1); h = g + tf(3,1);", "assert(strcmp(class(h), 'tf')); assert(isequal(h.Numerator, 5)); assert(isequal(h.Denominator, 1));");
method_documentation!(
    MINUS,
    "tf.minus",
    "Subtract one SISO tf model or scalar gain from another.",
    "g = tf(7,1); h = g - tf(3,1);",
    "assert(isequal(h.Numerator, 4)); assert(isequal(h.Denominator, 1));"
);
method_documentation!(
    UPLUS,
    "tf.uplus",
    "Return a value-semantic copy of a SISO tf model.",
    "g = tf(2,1); h = +g;",
    "assert(strcmp(class(h), 'tf')); assert(isequal(h.Numerator, 2));"
);
method_documentation!(
    UMINUS,
    "tf.uminus",
    "Negate a SISO tf model.",
    "g = tf(2,1); h = -g;",
    "assert(isequal(h.Numerator, -2)); assert(isequal(h.Denominator, 1));"
);
method_documentation!(
    TIMES,
    "tf.times",
    "Multiply SISO tf models or scalar gains with elementwise operator syntax.",
    "g = tf(2,1); h = g .* tf(3,1);",
    "assert(isequal(h.Numerator, 6)); assert(isequal(h.Denominator, 1));"
);
method_documentation!(
    MTIMES,
    "tf.mtimes",
    "Multiply SISO tf models or scalar gains with matrix operator syntax.",
    "g = tf(2,1); h = g * tf(3,1);",
    "assert(isequal(h.Numerator, 6)); assert(isequal(h.Denominator, 1));"
);
method_documentation!(
    RDIVIDE,
    "tf.rdivide",
    "Divide SISO tf models or scalar gains with elementwise operator syntax.",
    "g = tf(6,1); h = g ./ tf(2,1);",
    "assert(h.Numerator(1) / h.Denominator(1) == 3);"
);
method_documentation!(
    MRDIVIDE,
    "tf.mrdivide",
    "Divide SISO tf models or scalar gains with right matrix-division syntax.",
    "g = tf(6,1); h = g / tf(2,1);",
    "assert(h.Numerator(1) / h.Denominator(1) == 3);"
);
method_documentation!(
    LDIVIDE,
    "tf.ldivide",
    "Divide SISO tf models or scalar gains with left elementwise-division syntax.",
    r"g = tf(2,1); h = g .\ tf(6,1);",
    "assert(h.Numerator(1) / h.Denominator(1) == 3);"
);
method_documentation!(
    MLDIVIDE,
    "tf.mldivide",
    "Divide SISO tf models or scalar gains with left matrix-division syntax.",
    r"g = tf(2,1); h = g \ tf(6,1);",
    "assert(h.Numerator(1) / h.Denominator(1) == 3);"
);
method_documentation!(
    POWER,
    "tf.power",
    "Raise a SISO tf model to an integer power with elementwise operator syntax.",
    "g = tf(2,1); h = g .^ 3;",
    "assert(isequal(h.Numerator, 8)); assert(isequal(h.Denominator, 1));"
);
method_documentation!(
    MPOWER,
    "tf.mpower",
    "Raise a SISO tf model to an integer power with matrix operator syntax.",
    "g = tf(2,1); h = g ^ 3;",
    "assert(isequal(h.Numerator, 8)); assert(isequal(h.Denominator, 1));"
);

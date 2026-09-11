use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`idivide(A, B)` divides integer values and returns a quotient in the nondouble integer input class. Compatible dense shapes use implicit expansion.",
            "The default `fix` mode truncates toward zero. `floor` rounds toward negative infinity, `ceil` toward positive infinity, and `round` selects the nearest integer with halfway cases away from zero.",
            "A scalar `double` operand is accepted when the other operand has a non-64-bit integer class. Integer classes must otherwise match. Two-double calls, division by zero, incompatible shapes, invalid rounding modes, and unrepresentable quotients produce structured errors.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident values",
        paragraphs: &[
            "Resident integer operands must share one owning provider. They gather without changing class, and the result returns to that provider. The rounding mode is a host control and does not select residency.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "fix",
        title: "Truncate toward zero",
        program: "q = idivide(int16(-7), int16(3))",
        display_output: Some("q = int16(-2)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(q, 'int16')); assert(q == int16(-2));",
        },
    },
    BuiltinExample {
        id: "floor-array",
        title: "Select floor rounding",
        program: "q = idivide(uint16([9 10 11]), 3, 'floor')",
        display_output: Some("q = uint16([3 3 3])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(q, 'uint16')); assert(isequal(q, uint16([3 3 3])));",
        },
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Integer-division runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/integer_division",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "All integer classes, rounding modes, broadcasting, exact 64-bit values, errors, and resident fallback",
        location: "crates/runmat-runtime/src/builtins/math/integer_division/tests",
    }],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("idivide"),
    slug: Some("idivide"),
    summary: "Divide integer values with class-preserving rounded quotients.",
    description: "`idivide` divides integer values and applies a selected integral rounding rule while retaining the nondouble integer class.",
    keywords: &["idivide", "integer", "division", "rounding", "gpuArray"],
    related: &["fix", "floor", "ceil", "round"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "fix",
            target: BuiltinDocumentationLinkTarget::Builtin("fix"),
        },
        BuiltinDocumentationLink {
            label: "floor",
            target: BuiltinDocumentationLinkTarget::Builtin("floor"),
        },
    ],
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

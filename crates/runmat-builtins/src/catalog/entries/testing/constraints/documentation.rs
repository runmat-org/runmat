use crate::*;

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Constraint constructors and test qualification",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/constraints.rs"),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Constraint class, expected-value retention, and argument validation",
        location: "crates/runmat-runtime/src/builtins/testing/constraints.rs tests",
    }],
    notes: &["No editable legacy JSON sidecars exist for these three identities."],
};

const IS_EQUAL_TO_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "expected-value",
        title: "Retain an expected value in an equality constraint",
        program: "constraint = matlab.unittest.constraints.IsEqualTo(uint64(42));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(constraint, 'matlab.unittest.constraints.IsEqualTo'));\nassert(isa(constraint, 'matlab.unittest.constraints.Constraint'));\nassert(strcmp(class(constraint.Expected), 'uint64'));\nassert(isequal(constraint.Expected, uint64(42)));",
        },
    },
    BuiltinExample {
        id: "unsupported-options",
        title: "Unsupported comparison options fail during construction",
        program: "matlab.unittest.constraints.IsEqualTo('A', 'IgnoringCase', true);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:Testing:InvalidConstraint",
        },
    },
];

const IS_TRUE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "logical-true",
    title: "Create a logical-true constraint",
    program: "constraint = matlab.unittest.constraints.IsTrue();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(constraint, 'matlab.unittest.constraints.IsTrue'));\nassert(isa(constraint, 'matlab.unittest.constraints.Constraint'));",
    },
}];

const IS_FALSE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "logical-false",
    title: "Create a logical-false constraint",
    program: "constraint = matlab.unittest.constraints.IsFalse();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(constraint, 'matlab.unittest.constraints.IsFalse'));\nassert(isa(constraint, 'matlab.unittest.constraints.Constraint'));",
    },
}];

pub(super) const IS_EQUAL_TO: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.constraints.IsEqualTo"),
    slug: Some("matlab.unittest.constraints.IsEqualTo"),
    summary: "Construct an equality constraint for a test qualification.",
    description: "`IsEqualTo(expected)` retains `expected` in the constraint's public `Expected` property. RunMat currently supports this single positional form.",
    keywords: &["test", "constraint", "IsEqualTo", "equality"],
    related: &["matlab.unittest.constraints.IsTrue", "matlab.unittest.constraints.IsFalse"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported form",
            paragraphs: &["Pass exactly one expected value. The constructor retains its class and storage rather than converting it to double. The returned constraint object is host-resident; an expected value nested inside it can still hold a device or other non-transferable resource. RunMat does not promise worker transferability for that object."],
        },
        BuiltinDocumentationSection {
            heading: "Current comparison limits",
            paragraphs: &["MATLAB supports `IgnoringCase`, `IgnoringWhitespace`, `Using`, and `Within` options; these constructor forms are not implemented in RunMat and return `RunMat:Testing:InvalidConstraint`. RunMat's current `verifyThat`/`assertThat` equality path compares real numeric values with their class, shape, and exact integer payload intact, including NaNs. It remains a subset of MATLAB's comparator behavior for other value kinds: it does not invoke user-defined comparators or tolerance objects. Use `verifyEqual` for the separate qualification contract. The constraint is currently value-backed; no shared handle identity is promised."],
        },
    ],
    examples: IS_EQUAL_TO_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB IsEqualTo class",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.constraints.isequalto-class.html"),
    }],
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const IS_TRUE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.constraints.IsTrue"),
    slug: Some("matlab.unittest.constraints.IsTrue"),
    summary: "Construct a constraint satisfied by logical true.",
    description: "`IsTrue()` creates a constraint for a scalar logical `true` value. Numeric `1` and nonscalar logical arrays do not satisfy it.",
    keywords: &["test", "constraint", "IsTrue", "logical"],
    related: &["matlab.unittest.constraints.IsFalse", "matlab.unittest.constraints.IsEqualTo"],
    sections: &[BuiltinDocumentationSection {
        heading: "Value and placement",
        paragraphs: &["The constructor takes no arguments and returns a host scalar object. Satisfaction is checked by `verifyThat` or `assertThat`; constructing the object alone does not record a test result. The object is currently value-backed, so shared handle identity is not promised."],
    }],
    examples: IS_TRUE_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB IsTrue class",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.constraints.istrue-class.html"),
    }],
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const IS_FALSE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.constraints.IsFalse"),
    slug: Some("matlab.unittest.constraints.IsFalse"),
    summary: "Construct a constraint satisfied by logical false.",
    description: "`IsFalse()` creates a constraint for a scalar logical `false` value. Numeric `0` and nonscalar logical arrays do not satisfy it.",
    keywords: &["test", "constraint", "IsFalse", "logical"],
    related: &["matlab.unittest.constraints.IsTrue", "matlab.unittest.constraints.IsEqualTo"],
    sections: &[BuiltinDocumentationSection {
        heading: "Value and placement",
        paragraphs: &["The constructor takes no arguments and returns a host scalar object. Satisfaction is checked by `verifyThat` or `assertThat`; constructing the object alone does not record a test result. The object is currently value-backed, so shared handle identity is not promised."],
    }],
    examples: IS_FALSE_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB IsFalse class",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.constraints.isfalse-class.html"),
    }],
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

use crate::*;

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[
        BuiltinDocumentationLink {
            label: "Assertions",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/qualifications/assert.rs"),
        },
        BuiltinDocumentationLink {
            label: "Assumptions",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/qualifications/assume.rs"),
        },
        BuiltinDocumentationLink {
            label: "Shared qualification behavior",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/qualifications/shared.rs"),
        },
    ],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Qualification recording, abort, diagnostic, and constraint tests",
        location: "crates/runmat-runtime/src/builtins/testing/qualifications/shared.rs tests",
    }],
    notes: &["No editable legacy JSON sidecar exists for these five identities. Runtime qualifications currently implement a subset of MATLAB's comparison, diagnostic, and constraint behavior; the documented examples use RunMat's direct TestCase constructor."],
};

const ASSERT_EQUAL_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "assert-equal-pass",
        title: "Continue after an equality assertion passes",
        program: "testCase = matlab.unittest.TestCase();\nresult = assertEqual(testCase, 7, 7);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isempty(result));\nassert(strcmp(class(result), 'double'));",
        },
    },
    BuiltinExample {
        id: "assert-equal-fail",
        title: "Abort on unequal values",
        program: "testCase = matlab.unittest.TestCase();\nassertEqual(testCase, 7, 8);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:Test:AssertionFailed",
        },
    },
];

const ASSERT_THAT_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "assert-that-is-true",
        title: "Assert a built-in constraint",
        program: "testCase = matlab.unittest.TestCase();\nconstraint = matlab.unittest.constraints.IsTrue();\nresult = assertThat(testCase, true, constraint);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(result));" },
    },
    BuiltinExample {
        id: "assert-that-invalid-constraint",
        title: "Reject a non-constraint value",
        program: "testCase = matlab.unittest.TestCase();\nassertThat(testCase, true, 1);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:Testing:InvalidConstraint" },
    },
];

const ASSERT_TRUE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "assert-true-logical",
    title: "Assert a logical scalar",
    program: "testCase = matlab.unittest.TestCase();\nresult = assertTrue(testCase, true);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isempty(result));",
    },
}];

const ASSUME_EQUAL_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "assume-equal-pass",
    title: "Continue when an equality assumption holds",
    program:
        "testCase = matlab.unittest.TestCase();\nresult = assumeEqual(testCase, 'ready', 'ready');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isempty(result));",
    },
}];

const ASSUME_TRUE_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "assume-true-pass",
        title: "Continue when a logical assumption holds",
        program: "testCase = matlab.unittest.TestCase();\nresult = assumeTrue(testCase, true);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isempty(result));",
        },
    },
    BuiltinExample {
        id: "assume-true-fail-outside-test",
        title: "Report a failed assumption outside an active test",
        program: "testCase = matlab.unittest.TestCase();\nassumeTrue(testCase, false);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:Test:AssumptionFailed",
        },
    },
];

const EQUAL_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Equality and test lifecycle",
        paragraphs: &[
            "RunMat checks the receiver as a TestCase or subclass, then compares `actual` and `expected`. A passing call returns an empty double matrix. A failing assertion records a diagnostic in an active test and aborts that test; a failing assumption records a diagnostic and filters that test.",
            "This comparison is not yet the complete MATLAB equality contract. Scalar numeric comparisons currently convert to double, which can erase distinctions between wide integers. Additional MATLAB name-value tolerance options and complete custom-object comparison behavior are not implemented here.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Diagnostics",
        paragraphs: &["Optional diagnostic values are accepted after the required inputs. On failure, RunMat currently uses the first scalar string, character row, or Diagnostic object's `Message` as the failure message. Other diagnostic forms and full MATLAB diagnostic evaluation are not yet supported."],
    },
];

const LOGICAL_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Logical qualification",
        paragraphs: &["The current predicate passes for a scalar logical `true`. A nonlogical or nonscalar value fails the qualification rather than being converted to logical. A passing call returns an empty double matrix."],
    },
    BuiltinDocumentationSection {
        heading: "Test lifecycle and diagnostics",
        paragraphs: &["A failed assertion aborts the active test; a failed assumption filters it. Outside an active test, failure is returned as a RunMat error. Optional diagnostics currently use only the first scalar text or Diagnostic object's `Message`; complete MATLAB diagnostic behavior is not implemented."],
    },
];

pub(super) const ASSERT_EQUAL: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.assertEqual"),
    slug: Some("matlab.unittest.TestCase.assertEqual"),
    summary: "Assert equality and abort the current test on failure.",
    description: "`assertEqual(testCase, actual, expected)` qualifies equality. RunMat records a failed assertion in the active test and stops that test.",
    keywords: &["test", "assertion", "equality", "diagnostic"],
    related: &["matlab.unittest.TestCase.assumeEqual", "matlab.unittest.TestCase.assertThat"],
    sections: EQUAL_SECTIONS,
    examples: ASSERT_EQUAL_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB Assertable methods", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.assertable-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ASSERT_THAT: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.assertThat"),
    slug: Some("matlab.unittest.TestCase.assertThat"),
    summary: "Assert that a value satisfies a supported constraint.",
    description: "`assertThat(testCase, actual, constraint)` evaluates a supported Constraint object and aborts the active test if it is not satisfied.",
    keywords: &["test", "assertion", "constraint", "diagnostic"],
    related: &["matlab.unittest.constraints.IsEqualTo", "matlab.unittest.constraints.IsTrue", "matlab.unittest.constraints.IsFalse"],
    sections: &[
        BuiltinDocumentationSection { heading: "Supported constraints", paragraphs: &["RunMat currently evaluates `IsEqualTo`, `IsTrue`, and `IsFalse`. Compatible custom Constraint subclasses without a supported satisfaction hook return `RunMat:Testing:UnsupportedConstraint`; non-constraints return `RunMat:Testing:InvalidConstraint`. `IsEqualTo` uses the current RunMat constraint comparator and is not a claim of complete MATLAB comparator parity."] },
        BuiltinDocumentationSection { heading: "Result and diagnostics", paragraphs: &["A passing call returns an empty double matrix. A failed assertion records a diagnostic and aborts the active test. Only the first supported optional diagnostic supplies a custom message."] },
    ],
    examples: ASSERT_THAT_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB Assertable methods", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.assertable-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ASSERT_TRUE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.assertTrue"),
    slug: Some("matlab.unittest.TestCase.assertTrue"),
    summary: "Assert logical truth and abort the current test on failure.",
    description: "`assertTrue(testCase, actual)` passes only for a scalar logical true value in the current runtime.",
    keywords: &["test", "assertion", "logical", "diagnostic"],
    related: &["matlab.unittest.TestCase.assumeTrue", "matlab.unittest.constraints.IsTrue"],
    sections: LOGICAL_SECTIONS,
    examples: ASSERT_TRUE_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB Assertable methods", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.assertable-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ASSUME_EQUAL: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.assumeEqual"),
    slug: Some("matlab.unittest.TestCase.assumeEqual"),
    summary: "Require equality or filter the current test.",
    description: "`assumeEqual(testCase, actual, expected)` qualifies a test precondition. A failed assumption filters the active test rather than reporting an assertion failure.",
    keywords: &["test", "assumption", "equality", "diagnostic"],
    related: &["matlab.unittest.TestCase.assertEqual"],
    sections: EQUAL_SECTIONS,
    examples: ASSUME_EQUAL_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB Assumable methods", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.assumable-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ASSUME_TRUE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.assumeTrue"),
    slug: Some("matlab.unittest.TestCase.assumeTrue"),
    summary: "Require logical truth or filter the current test.",
    description: "`assumeTrue(testCase, actual)` passes for scalar logical true and filters the active test otherwise.",
    keywords: &["test", "assumption", "logical", "diagnostic"],
    related: &["matlab.unittest.TestCase.assertTrue"],
    sections: LOGICAL_SECTIONS,
    examples: ASSUME_TRUE_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB Assumable methods", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.assumable-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

use crate::*;

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[
        BuiltinDocumentationLink { label: "Verification builtins", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/qualifications/verify.rs") },
        BuiltinDocumentationLink { label: "Shared qualification predicates and recording", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/qualifications/shared.rs") },
    ],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Verification recording, continuation, and comparison tests", location: "crates/runmat-runtime/src/builtins/testing/qualifications/shared.rs tests" }],
    notes: &["No editable legacy JSON sidecar exists for these four identities. The current comparison and diagnostic implementations cover a subset of MATLAB behavior; RunMat's direct TestCase constructor is used only in RunMat-compatible examples."],
};

const LIFECYCLE: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading: "Verification lifecycle",
    paragraphs: &["In an active test, a failed verification records a diagnostic and returns an empty double matrix so that the test continues. Outside an active test, a failed verification returns `RunMat:Test:VerificationFailed`. Passing calls also return an empty double matrix. Optional diagnostics currently use only the first scalar text or Diagnostic object's `Message`."],
};

const EMPTY_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "verify-empty-array",
    title: "Verify an empty array",
    program: "testCase = matlab.unittest.TestCase();\nresult = verifyEmpty(testCase, []);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isempty(result));",
    },
}];

const EQUAL_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "verify-equal-pass",
        title: "Verify equal values",
        program: "testCase = matlab.unittest.TestCase();\nresult = verifyEqual(testCase, [1, 2], [1, 2]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(result));" },
    },
    BuiltinExample {
        id: "verify-equal-outside-test",
        title: "Report failed verification outside a test",
        program: "testCase = matlab.unittest.TestCase();\nverifyEqual(testCase, 1, 2);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:Test:VerificationFailed" },
    },
];

const FALSE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "verify-false-logical",
    title: "Verify logical false",
    program: "testCase = matlab.unittest.TestCase();\nresult = verifyFalse(testCase, false);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isempty(result));",
    },
}];

const GREATER_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "verify-greater-than-scalars",
    title: "Verify a scalar exceeds a threshold",
    program: "testCase = matlab.unittest.TestCase();\nresult = verifyGreaterThan(testCase, 5, 3);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isempty(result));",
    },
}];

pub(super) const VERIFY_EMPTY: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.verifyEmpty"),
    slug: Some("matlab.unittest.TestCase.verifyEmpty"),
    summary: "Verify that a value has no elements.",
    description: "`verifyEmpty(testCase, actual)` checks whether the runtime value has zero elements.",
    keywords: &["test", "verification", "empty", "diagnostic"],
    related: &["matlab.unittest.TestCase.verifyNotEmpty", "isempty"],
    sections: &[
        BuiltinDocumentationSection { heading: "Empty-value check", paragraphs: &["The current implementation checks the value's element count. It may request runtime shape information before recording the result. It does not assert that the input has a particular class or shape beyond zero elements."] },
        LIFECYCLE,
    ],
    examples: EMPTY_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB verifyEmpty", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.verifiable.verifyempty.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const VERIFY_EQUAL: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.verifyEqual"),
    slug: Some("matlab.unittest.TestCase.verifyEqual"),
    summary: "Verify equality without aborting the current test.",
    description: "`verifyEqual(testCase, actual, expected)` compares values and records a soft verification failure if they differ.",
    keywords: &["test", "verification", "equality", "diagnostic"],
    related: &["matlab.unittest.TestCase.assertEqual", "matlab.unittest.TestCase.verifyNotEqual"],
    sections: &[
        BuiltinDocumentationSection { heading: "Current comparison limits", paragraphs: &["Scalar numeric comparisons currently pass through double, so distinct wide integer values can compare equal. MATLAB name-value tolerance options such as `AbsTol` and `RelTol`, and full custom-object comparison behavior, are not implemented. The comparison of other values follows the current RunMat value equality implementation."] },
        LIFECYCLE,
    ],
    examples: EQUAL_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB verifyEqual", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.verifiable.verifyequal.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const VERIFY_FALSE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.verifyFalse"),
    slug: Some("matlab.unittest.TestCase.verifyFalse"),
    summary: "Verify that a value is logical false.",
    description: "`verifyFalse(testCase, actual)` passes only for a scalar logical false value in the current runtime.",
    keywords: &["test", "verification", "logical", "diagnostic"],
    related: &["matlab.unittest.TestCase.verifyTrue", "matlab.unittest.constraints.IsFalse"],
    sections: &[
        BuiltinDocumentationSection { heading: "Logical predicate", paragraphs: &["Nonlogical values and nonscalar logical arrays fail the qualification rather than being implicitly converted to logical false."] },
        LIFECYCLE,
    ],
    examples: FALSE_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB verifyFalse", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.verifiable.verifyfalse.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const VERIFY_GREATER_THAN: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.verifyGreaterThan"),
    slug: Some("matlab.unittest.TestCase.verifyGreaterThan"),
    summary: "Verify that a numeric scalar exceeds a threshold.",
    description: "`verifyGreaterThan(testCase, actual, floor)` compares two numeric scalar values in the current RunMat implementation.",
    keywords: &["test", "verification", "comparison", "diagnostic"],
    related: &["matlab.unittest.TestCase.verifyLessThan", "matlab.unittest.TestCase.verifyEqual"],
    sections: &[
        BuiltinDocumentationSection { heading: "Current comparison limits", paragraphs: &["RunMat currently compares numeric scalars after conversion to double. This does not preserve ordering for every wide integer and does not implement MATLAB's all-elements array comparison. Other kinds fail the qualification rather than being accepted as comparable numeric scalars."] },
        LIFECYCLE,
    ],
    examples: GREATER_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB verifyGreaterThan", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.verifiable.verifygreaterthan.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

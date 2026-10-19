use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "fatal-assert-true-pass",
        title: "Continue when a fatal assertion passes",
        program:
            "testCase = matlab.unittest.TestCase();\nresult = fatalAssertTrue(testCase, true);",
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
        id: "fatal-assert-true-outside-test",
        title: "Report a failed fatal assertion outside a test run",
        program: "testCase = matlab.unittest.TestCase();\nfatalAssertTrue(testCase, false);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:Test:FatalAssertionFailed",
        },
    },
];

pub(super) const FATAL_ASSERT_TRUE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.fatalAssertTrue"),
    slug: Some("matlab.unittest.TestCase.fatalAssertTrue"),
    summary: "Require logical truth or abort the current test run.",
    description: "`fatalAssertTrue(testCase, actual)` passes for a scalar logical true value. On failure, RunMat records a fatal diagnostic and aborts the active run.",
    keywords: &["test", "fatal assertion", "logical", "diagnostic"],
    related: &["matlab.unittest.TestCase.assertTrue", "matlab.unittest.TestCase.verifyFalse"],
    sections: &[
        BuiltinDocumentationSection { heading: "Predicate and result", paragraphs: &["Only a scalar logical true value passes in the current runtime. Nonlogical and nonscalar values fail the qualification instead of being implicitly converted. A passing call returns an empty double matrix."] },
        BuiltinDocumentationSection { heading: "Failure and lifecycle", paragraphs: &["Within an active test, a failure records a `FatalAssertionFailed` qualification and requests run abort. The testing executor is responsible for the remaining teardown lifecycle. Outside an active test, the failure is returned as `RunMat:Test:FatalAssertionFailed`. Optional diagnostic values currently use only the first supported scalar text or Diagnostic object's `Message`."] },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB FatalAssertable methods", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.qualifications.fatalassertable-class.html") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Fatal qualification and shared lifecycle", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/qualifications/fatal.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Qualification failure recording and abort behavior", location: "crates/runmat-runtime/src/builtins/testing/qualifications/shared.rs tests" }],
        notes: &["No editable legacy JSON sidecar exists. The direct TestCase constructor used in these RunMat examples is not a MATLAB-compatible way to instantiate MATLAB's abstract TestCase class."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

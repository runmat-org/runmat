use crate::*;

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Testing diagnostic builtins",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/qualifications/diagnostics.rs"),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Diagnostic constructors and test-context recording",
        location: "crates/runmat-runtime/src/builtins/testing/qualifications/diagnostics.rs tests",
    }],
    notes: &["No editable legacy JSON sidecar exists for these three identities."],
};

const TEST_CASE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "construct-test-case",
    title: "Create a test case for RunMat testing utilities",
    program: "testCase = matlab.unittest.TestCase();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(testCase, 'matlab.unittest.TestCase'));",
    },
}];

const STRING_DIAGNOSTIC_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "diagnostic-text",
    title: "Create a diagnostic and inspect its message",
    program: "diagnostic = matlab.unittest.diagnostics.StringDiagnostic('ready');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(diagnostic, 'matlab.unittest.diagnostics.StringDiagnostic'));\nassert(strcmp(diagnostic.DiagnosticText, 'ready'));",
    },
}];

const LOG_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "requires-active-test",
    title: "Logging outside a test reports the missing lifecycle context",
    program: "testCase = matlab.unittest.TestCase();\nlog(testCase, 'starting');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::ExpectedError {
        identifier: "RunMat:Testing:InvalidDiagnostic",
    },
}];

pub(super) const TEST_CASE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase"), slug: Some("matlab.unittest.TestCase"),
    summary: "Create a RunMat test-case object.",
    description: "`matlab.unittest.TestCase()` creates a lightweight test-case object in RunMat. Function-based tests receive their own case object from the test runner.",
    keywords: &["test", "TestCase", "testing", "constructor"],
    related: &["matlab.unittest.TestCase.log"],
    sections: &[BuiltinDocumentationSection {
        heading: "Usage",
        paragraphs: &["The direct zero-argument constructor is a RunMat convenience form. It creates an object with an empty `Name`; it does not start a test lifecycle. Qualification and logging methods that record results require an active test run.", "RunMat currently stores this object by value. Its registered class inherits from `handle` for class and method lookup, but the object does not yet preserve handle identity or shared mutation across copies. MATLAB test classes normally derive from `matlab.unittest.TestCase`, or obtain a case object through its test framework; the direct constructor is not a substitute for a test runner."],
    }],
    examples: TEST_CASE_EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB TestCase class", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.testcase-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const STRING_DIAGNOSTIC: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.diagnostics.StringDiagnostic"), slug: Some("matlab.unittest.diagnostics.StringDiagnostic"),
    summary: "Wrap scalar text as a test diagnostic.",
    description: "`StringDiagnostic(text)` creates a diagnostic object with `DiagnosticText` and `Message` set to the supplied text.",
    keywords: &["test", "diagnostic", "StringDiagnostic", "message"],
    related: &["matlab.unittest.TestCase.log", "matlab.unittest.TestCase"],
    sections: &[BuiltinDocumentationSection {
        heading: "Input and class",
        paragraphs: &["RunMat currently accepts one host string scalar or character row. Numeric values, text arrays and GPU-resident text are rejected.", "The result retains the concrete `matlab.unittest.diagnostics.StringDiagnostic` class and is a subclass of RunMat's `Diagnostic` class. It is currently stored as a value-backed object; handle identity and shared mutation are not provided. MATLAB has also exposed this class under `matlab.automation.diagnostics` since R2023a; that newer constructor name is not yet registered in RunMat."],
    }],
    examples: STRING_DIAGNOSTIC_EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB StringDiagnostic class", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.automation.diagnostics.stringdiagnostic-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const LOG: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.log"), slug: Some("matlab.unittest.TestCase.log"),
    summary: "Record a diagnostic in the active test result.",
    description: "`log(testCase, diagnostic)` records at default verbosity 2; `log(testCase, verbosity, diagnostic)` selects verbosity 1 through 4 or a named level.",
    keywords: &["test", "log", "diagnostic", "verbosity"],
    related: &["matlab.unittest.TestCase", "matlab.unittest.diagnostics.StringDiagnostic"],
    sections: &[
        BuiltinDocumentationSection { heading: "Test context", paragraphs: &["The call records a structured diagnostic in the current test lifecycle. Calling it outside an active test returns `RunMat:Testing:InvalidDiagnostic`; it does not silently discard the message.", "A successful call returns an empty double matrix. It does not itself fail the test. RunMat stores the selected verbosity in the diagnostic details; the test-result renderer determines whether and where to display it."] },
        BuiltinDocumentationSection { heading: "Accepted values", paragraphs: &["Verbosity defaults to `Concise` (2). Accepted explicit values are numeric scalars 1 through 4, including one-element numeric arrays, or the names `Terse`, `Concise`, `Detailed`, and `Verbose` (case-insensitive). The `matlab.unittest.Verbosity` enumeration is not implemented. Diagnostics may be scalar text or a `Diagnostic` object. Function-handle and diagnostic-array inputs from MATLAB's wider API are not yet supported."] },
    ],
    examples: LOG_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB TestCase.log method", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.testcase.log.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

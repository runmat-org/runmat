use crate::*;

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[
        BuiltinDocumentationLink { label: "Fixture constructors and application", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/fixtures/fixture.rs") },
        BuiltinDocumentationLink { label: "Teardown registration", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/fixtures/teardown.rs") },
    ],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Fixture lifecycle and teardown context tests",
        location: "crates/runmat-runtime/src/builtins/testing/fixtures/teardown.rs tests; crates/runmat-runtime/src/testing/context.rs tests",
    }],
    notes: &["No editable legacy JSON sidecar exists for these four identities. These APIs are partial until fixture options, path restoration, and complete test lifecycle parity are implemented and verified."],
};

const PATH_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "construct-path-fixture", title: "Create a path fixture without applying it",
    program: "fixture = matlab.unittest.fixtures.PathFixture('src');",
    display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(fixture, 'matlab.unittest.fixtures.PathFixture'));\nassert(strcmp(fixture.Path, 'src'));",
    },
}];
const TEMP_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "construct-temporary-folder-fixture", title: "Create a temporary-folder fixture before setup",
    program: "fixture = matlab.unittest.fixtures.TemporaryFolderFixture();",
    display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(fixture, 'matlab.unittest.fixtures.TemporaryFolderFixture'));\nassert(strcmp(fixture.Folder, ''));",
    },
}];
const APPLY_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "requires-active-test-for-fixture", title: "Reject fixture application outside a test",
    program: "testCase = matlab.unittest.TestCase();\nfixture = matlab.unittest.fixtures.TemporaryFolderFixture();\napplyFixture(testCase, fixture);",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:Testing:InvalidFixture" },
}];
const TEARDOWN_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "requires-active-test-for-teardown",
    title: "Reject teardown registration outside a test",
    program: "testCase = matlab.unittest.TestCase();\naddTeardown(testCase, @disp, 'done');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::ExpectedError {
        identifier: "RunMat:Testing:AddTeardown",
    },
}];

pub(super) const PATH_FIXTURE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.fixtures.PathFixture"), slug: Some("matlab.unittest.fixtures.PathFixture"),
    summary: "Create a fixture for a search-path folder.",
    description: "`PathFixture(folder)` creates a handle fixture with its `Path` property. Construction does not change the session path; applying the fixture does.",
    keywords: &["test", "fixture", "path", "search path"],
    related: &["addpath", "rmpath", "matlab.unittest.TestCase.applyFixture"],
    sections: &[BuiltinDocumentationSection { heading: "Current support", paragraphs: &[
        "RunMat accepts one host string scalar or character row. MATLAB also supports multiple folders and `IncludingSubfolders`/`Position` options; these are not implemented here.",
        "The fixture is a handle object. In an active test, `applyFixture` currently adds the folder and registers `rmpath` for teardown. Exact restoration of a pre-existing path entry and its ordering is not yet provided.",
    ] }],
    examples: PATH_EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB PathFixture", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.fixtures.pathfixture-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const TEMPORARY_FOLDER_FIXTURE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.fixtures.TemporaryFolderFixture"), slug: Some("matlab.unittest.fixtures.TemporaryFolderFixture"),
    summary: "Create a fixture for a test-scoped temporary folder.",
    description: "`TemporaryFolderFixture()` creates a handle fixture. When applied in a test, RunMat creates its folder, sets `Folder`, and registers recursive removal at teardown.",
    keywords: &["test", "fixture", "temporary folder", "filesystem"],
    related: &["matlab.unittest.TestCase.applyFixture", "matlab.unittest.TestCase.addTeardown"],
    sections: &[BuiltinDocumentationSection { heading: "Current support", paragraphs: &[
        "RunMat currently accepts the zero-argument constructor only. MATLAB's `WithSuffix` and `PreservingOnFailure` options are not implemented.",
        "The current folder is under RunMat's test-tmp namespace and is not yet guaranteed to be an absolute path. `Folder` is empty until setup. Cleanup uses the test teardown lifecycle; explicit failure-preservation policy is not available.",
    ] }],
    examples: TEMP_EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB TemporaryFolderFixture", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.fixtures.temporaryfolderfixture-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const APPLY_FIXTURE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.applyFixture"), slug: Some("matlab.unittest.TestCase.applyFixture"),
    summary: "Apply a fixture within an active test lifecycle.",
    description: "`applyFixture(testCase, fixture)` sets up a supported fixture and returns the same handle after setup. A teardown command is registered for the active test scope.",
    keywords: &["test", "fixture", "setup", "teardown"],
    related: &["matlab.unittest.fixtures.PathFixture", "matlab.unittest.fixtures.TemporaryFolderFixture"],
    sections: &[BuiltinDocumentationSection { heading: "Lifecycle and limits", paragraphs: &[
        "An active test lifecycle is required and checked before path or filesystem mutation. PathFixture and TemporaryFolderFixture have concrete setup behavior; arbitrary Fixture subclasses are not yet given user-defined setup and teardown dispatch.",
        "Fixture teardown is recorded in the current test context. Complete exception-safe rollback after setup/registration failure, custom fixture hooks, and full MATLAB fixture-scope parity still require work.",
    ] }],
    examples: APPLY_EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB applyFixture", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.testcase.applyfixture.html") }],
    media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ADD_TEARDOWN: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.TestCase.addTeardown"), slug: Some("matlab.unittest.TestCase.addTeardown"),
    summary: "Register a callback for test teardown.",
    description: "`addTeardown(testCase, callback, args...)` records a function handle and its arguments in the active test context. The test executor invokes it during teardown.",
    keywords: &["test", "teardown", "callback", "cleanup"],
    related: &["matlab.unittest.TestCase.applyFixture"],
    sections: &[BuiltinDocumentationSection { heading: "Lifecycle and callback", paragraphs: &[
        "Registration requires an active non-teardown test phase. The callback and arguments are retained for later execution; their effects cannot be statically bounded when the handle is unknown.",
        "RunMat records a structured teardown command and a runtime invocation. User-function closures and method handles have not yet been fully qualified for capture and invocation parity across all execution targets; this contract remains partial.",
    ] }],
    examples: TEARDOWN_EXAMPLES, example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB addTeardown", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.testcase.addteardown.html") }],
    media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

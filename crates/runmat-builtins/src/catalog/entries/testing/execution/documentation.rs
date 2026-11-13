use crate::*;

const TEST_FILE: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "function-test-outcomes" },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File {
        relative_path: "sampleTest.m",
        content: BuiltinFixtureContent::Utf8("function tests = sampleTest()\n    tests = functiontests(localfunctions);\nend\n\nfunction testPass(testCase)\n    testCase.verifyEqual(2 + 2, 4);\nend\n\nfunction testFail(testCase)\n    testCase.verifyEqual(2 + 2, 5);\nend\n"),
    }],
});

const SUITE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "discover-local-tests", title: "Build a suite from local function handles",
    program: "suite = sampleTest();",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: TEST_FILE,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(numel(suite) == 2);\nassert(strcmp(suite(1).ProcedureName, 'testPass'));\nassert(strcmp(suite(2).ProcedureName, 'testFail'));" },
}];

const RUNNER_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "create-runner",
    title: "Construct a test runner",
    program: "runner = matlab.unittest.TestRunner.withTextOutput();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source:
            "assert(isa(runner, 'matlab.unittest.TestRunner'));\nassert(isempty(runner.Plugins));",
    },
}];

const PLUGIN_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "attach-plugin", title: "Attach a plugin to the same runner handle",
    program: "runner = matlab.unittest.TestRunner.withTextOutput();\nplugin = matlab.unittest.plugins.TestRunnerPlugin();\nrunner = runner.addPlugin(plugin);",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(isa(runner, 'matlab.unittest.TestRunner'));\nassert(numel(runner.Plugins) == 1);" },
}];

const RUN_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "inspect-pass-and-fail", title: "Inspect both successful and failed test outcomes",
    program: "suite = sampleTest();\nrunner = matlab.unittest.TestRunner.withTextOutput();\nresults = runner.run(suite);",
    display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem, fixture: TEST_FILE,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(numel(results) == 2);\nassert(results(1).Passed);\nassert(~results(1).Failed);\nassert(results(2).Failed);\nassert(~results(2).Passed);\nassert(~results(2).Incomplete);" },
}];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[
        BuiltinDocumentationLink { label: "Function-based suite construction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/functiontests.rs") },
        BuiltinDocumentationLink { label: "Runner construction and plugin storage", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/testrunner.rs") },
        BuiltinDocumentationLink { label: "Core test execution and TestResult projection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-core/src/testing/runtime_adapter.rs") },
    ],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Function handle filtering and fixture attachment", location: "crates/runmat-runtime/src/builtins/testing/functiontests.rs tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Function-test execution and structured results", location: "crates/runmat-core/src/testing/run.rs tests" },
    ],
    notes: &["No editable legacy JSON sidecars exist for these identities. The examples' test file has one deliberate qualification failure; its TestResult must report Failed rather than being swallowed as a green test. Runner options, plugin callbacks and MATLAB text reporter parity remain incomplete."],
};

pub(super) const FUNCTIONTESTS: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("functiontests"), slug: Some("functiontests"),
    summary: "Create a TestSuite from local function handles.",
    description: "`functiontests(localfunctions)` selects named local functions whose names begin with `test` and creates TestSuite objects; conventional setup and teardown handles are attached to each test.",
    keywords: &["test", "localfunctions", "function test", "suite"], related: &["matlab.unittest.TestRunner.run"],
    sections: &[BuiltinDocumentationSection { heading: "Current selection and fixtures", paragraphs: &["Pass a cell array of named function handles. The current implementation recognizes names beginning with `test`, case-insensitively, and `setupOnce`, `teardownOnce`, `setup`, and `teardown` fixtures. It does not yet select test names ending in `test`, which MATLAB also accepts.", "The returned suite records the source file, procedure names and fixture handles. Constructing the suite does not execute the tests; a Core-backed runner resolves the source and records structured outcomes."] }],
    examples: SUITE_EXAMPLES, example_exemption: None, faqs: &[], links: &[BuiltinDocumentationLink { label: "MATLAB functiontests", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/functiontests.html") }], media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const WITH_TEXT_OUTPUT: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("matlab.unittest.TestRunner.withTextOutput"), slug: Some("matlab.unittest.TestRunner.withTextOutput"),
    summary: "Construct a TestRunner handle for Core-backed tests.",
    description: "`withTextOutput()` creates a TestRunner handle with an empty plugin list. It can run a TestSuite through the active Core test executor.",
    keywords: &["test", "runner", "text output"], related: &["matlab.unittest.TestRunner.run", "matlab.unittest.TestRunner.addPlugin"],
    sections: &[BuiltinDocumentationSection { heading: "Current options and reporting", paragraphs: &["The supported form takes no arguments. An odd argument count returns InvalidRunner; any complete name-value pair returns UnsupportedRunnerOption instead of being silently ignored. MATLAB-compatible text progress and diagnostic output are not guaranteed yet. The returned handle retains plugins added later."] }],
    examples: RUNNER_EXAMPLES, example_exemption: None, faqs: &[], links: &[BuiltinDocumentationLink { label: "MATLAB withTextOutput", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.testrunner.withtextoutput.html") }], media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ADD_PLUGIN: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("matlab.unittest.TestRunner.addPlugin"), slug: Some("matlab.unittest.TestRunner.addPlugin"),
    summary: "Attach a TestRunnerPlugin to a runner handle.",
    description: "`runner.addPlugin(plugin)` mutates the runner's plugin list and returns the same TestRunner handle.",
    keywords: &["test", "runner", "plugin"], related: &["matlab.unittest.plugins.TestRunnerPlugin", "matlab.unittest.plugins.CodeCoveragePlugin.forFolder", "matlab.unittest.TestRunner.run"],
    sections: &[BuiltinDocumentationSection { heading: "Plugin execution", paragraphs: &["The plugin must be a TestRunnerPlugin object or subclass. The Core executor recognizes CodeCoveragePlugin folder selection during a run. Arbitrary plugin lifecycle callbacks and report generation are not dispatched yet; storing a custom plugin is not a promise that its hooks execute."] }],
    examples: PLUGIN_EXAMPLES, example_exemption: None, faqs: &[], links: &[BuiltinDocumentationLink { label: "MATLAB addPlugin", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.testrunner.addplugin.html") }], media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const RUN: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("matlab.unittest.TestRunner.run"), slug: Some("matlab.unittest.TestRunner.run"),
    summary: "Run selected tests and return structured TestResult objects.",
    description: "`runner.run(suite)` delegates a TestSuite selection to the active Core test executor and returns one TestResult per executed test.",
    keywords: &["test", "runner", "results", "failure"], related: &["functiontests", "matlab.unittest.TestRunner.withTextOutput", "runtests"],
    sections: &[BuiltinDocumentationSection { heading: "Execution and outcomes", paragraphs: &["The runner resolves the suite's source files through Core, executes selected tests in isolated test sessions and projects Passed, Failed, Incomplete, Details and other result properties. A failed test is represented by a failed TestResult; it does not make the runner call itself succeed as a passing test.", "This operation requires an installed Core testing service and access to the selected source files. Without one, it returns RunMat:Testing:RequiresExecutor. The exact MATLAB text reporter and arbitrary plugin callbacks are not implemented."] }],
    examples: RUN_EXAMPLES, example_exemption: None, faqs: &[], links: &[BuiltinDocumentationLink { label: "MATLAB TestRunner.run", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.testrunner.run.html") }], media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

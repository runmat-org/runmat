use crate::*;

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[
        BuiltinDocumentationLink { label: "TestRunnerPlugin constructor", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/plugins/test_runner_plugin.rs") },
        BuiltinDocumentationLink { label: "CodeCoveragePlugin.forFolder constructor", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/plugins/code_coverage_plugin.rs") },
        BuiltinDocumentationLink { label: "Core coverage selection and test-result projection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-core/src/testing/runtime_adapter.rs") },
    ],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Base plugin construction and invalid arity", location: "crates/runmat-runtime/src/builtins/testing/plugins/test_runner_plugin.rs tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Coverage plugin option and folder retention", location: "crates/runmat-runtime/src/builtins/testing/plugins/code_coverage_plugin.rs tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Coverage selection and filtering", location: "crates/runmat-core/src/testing/runtime_adapter.rs tests" },
    ],
    notes: &["No editable legacy JSON sidecar exists for these identities. MATLAB's plugin classes have handle semantics, while these current RunMat constructors return value-backed objects; complete custom plugin hooks and MATLAB report formats remain outside the implemented subset."],
};

const BASE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "construct-runner-plugin",
    title: "Construct a base plugin object",
    program: "plugin = matlab.unittest.plugins.TestRunnerPlugin();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(plugin, 'matlab.unittest.plugins.TestRunnerPlugin'));",
    },
}];

const COVERAGE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "configure-coverage-folders",
    title: "Configure coverage collection for a folder tree",
    program: "plugin = matlab.unittest.plugins.CodeCoveragePlugin.forFolder('src', 'IncludingSubfolders', true);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(isa(plugin, 'matlab.unittest.plugins.CodeCoveragePlugin'));\nassert(numel(plugin.Folders) == 1);\nassert(strcmp(plugin.Folders(1), 'src'));\nassert(plugin.IncludingSubfolders);" },
}];

pub(super) const TEST_RUNNER_PLUGIN: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.plugins.TestRunnerPlugin"),
    slug: Some("matlab.unittest.plugins.TestRunnerPlugin"),
    summary: "Construct a base test-runner plugin object.",
    description: "`TestRunnerPlugin()` creates a scalar plugin object accepted by `TestRunner.addPlugin`.",
    keywords: &["test", "runner", "plugin", "extension"],
    related: &["matlab.unittest.TestRunner.addPlugin", "matlab.unittest.plugins.CodeCoveragePlugin.forFolder"],
    sections: &[
        BuiltinDocumentationSection { heading: "Current plugin behavior", paragraphs: &["The constructor accepts no inputs and returns a host object of the concrete TestRunnerPlugin class. The current object is value-backed; RunMat does not yet promise MATLAB handle identity for base or custom plugins.", "A runner retains the object when `addPlugin` is called. Custom TestRunnerPlugin callback hooks are not dispatched by the current Core test executor, so constructing or adding a subclass is not a claim of MATLAB plugin extension parity."] },
    ],
    examples: BASE_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB TestRunnerPlugin class", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.plugins.testrunnerplugin-class.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const CODE_COVERAGE_FOR_FOLDER: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.unittest.plugins.CodeCoveragePlugin.forFolder"),
    slug: Some("matlab.unittest.plugins.CodeCoveragePlugin.forFolder"),
    summary: "Select source folders for test-run coverage collection.",
    description: "`CodeCoveragePlugin.forFolder(folder, ...)` creates a plugin that retains the selected folders and `IncludingSubfolders` flag for a RunMat TestRunner.",
    keywords: &["test", "coverage", "plugin", "folder"],
    related: &["matlab.unittest.TestRunner.addPlugin", "matlab.unittest.TestRunner.run"],
    sections: &[
        BuiltinDocumentationSection { heading: "Accepted constructor forms", paragraphs: &["Pass one or more nonempty host string scalars, character rows, or string arrays as folders. The first argument is always a folder. The supported name-value option after it is `IncludingSubfolders` as a string scalar or character row, with scalar logical true/false or numeric double 0/1. The constructor records paths without checking that the folders exist; empty entries in a string array are currently skipped.", "`Producing`, `MetricLevel`, and `Filter` are recognized as unsupported option names after the first folder and return UnsupportedPluginOption rather than silently becoming folders. Cell arrays of character vectors and coverage-report format objects are not implemented. Other unrecognized text arguments are treated as folders, so explicit option-like text that is not one of these names is not interpreted as an option."] },
        BuiltinDocumentationSection { heading: "Coverage during a test run", paragraphs: &["Construction alone does not run tests or instrument code. When the plugin is attached to a runner, the Core test executor uses its folder selection to filter collected coverage sites and projects a RunMat coverage summary onto test results. It does not emit MATLAB's HTML report or guarantee MATLAB coverage metrics and report formats."] },
    ],
    examples: COVERAGE_EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "MATLAB CodeCoveragePlugin.forFolder", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.unittest.plugins.codecoverageplugin.forfolder.html") }],
    media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

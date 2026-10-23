use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "discover-without-running",
        title: "Discover a script without executing its body",
        program: "suite = testsuite('testDiscovery.m');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "discover-without-running" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File {
                relative_path: "testDiscovery.m",
                content: BuiltinFixtureContent::Utf8("error('test body must not run during discovery');\n"),
            }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(numel(suite) == 1); assert(contains(suite.Name, 'testDiscovery'));",
        },
    },
    BuiltinExample {
        id: "filter-functions",
        title: "Select a semantic test procedure by name",
        program: "suite = testsuite('selectionTests.m', 'Name', 'testBeta');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "filter-functions" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File {
                relative_path: "selectionTests.m",
                content: BuiltinFixtureContent::Utf8("function tests = selectionTests()\n tests = functiontests(localfunctions);\nend\nfunction testAlpha(testCase)\n testCase.verifyFail('not selected');\nend\nfunction testBeta(testCase)\n testCase.verifyTrue(true);\nend\n"),
            }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(numel(suite) == 1); assert(contains(suite.Name, 'testBeta'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("testsuite"), slug: Some("testsuite"),
    summary: "Discover MATLAB-style test procedures without running their bodies.",
    description: "`testsuite` freezes selected source files, asks Core for semantic test discovery, and returns a scalar `matlab.unittest.TestSuite`-compatible object or a homogeneous object row. Discovery shares the same source and procedure identities as `runtests` and `TestRunner.run`; it does not execute test bodies.",
    keywords: &["testsuite", "testing", "discovery", "matlab.unittest"],
    related: &["runtests", "functiontests"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Targets and selection",
            paragraphs: &[
                "`suite = testsuite` discovers the current folder. `testsuite(target, Name, Value, ...)` accepts a test file, folder, bare function name, string array, or cell of scalar text. `BaseFolder` constrains relative explicit targets, or supplies discovery roots when no target is given.",
                "Folder discovery recognizes `.m` stems beginning with `test` or ending with `Test`, case-insensitively. `IncludeSubfolders=true` recursively searches subfolders. Explicit test paths need not match that naming convention.",
                "Core discovers script, function, `functiontests(localfunctions)`, and class suites semantically. `Name` and `ProcedureName` are substring filters on semantic display names, not the full MATLAB selector model. Runtime does not scan function declarations or append calls to source.",
                "The returned objects include `Name` and `TestFile` metadata and can be supplied to the existing TestRunner service. Discovery itself neither executes qualifications nor mutates the caller workspace.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Controls and limitations",
            paragraphs: &[
                "Host string scalars, character rows, and text collections control discovery. Exact integer, double, and logical flags accept only scalar zero or one. Resident controls reject without a GPU gather or provider lookup.",
                "`UseParallel=false` is accepted; true is unsupported by the in-program adapter. Nonempty `Tag` filters remain unsupported at this entry point. `OutputDetail` and `LoggingLevel` accept scalar text but do not alter discovery output. The RunMat `Coverage` flag is accepted by the shared parser but does not collect coverage during discovery.",
                "An active Core test service is required. It is selected from the explicitly retained RuntimeContext; no unrelated active session or alternate executor is substituted. Core constructs its existing child session and freezes the exact once-read source bytes.",
                "Directory discovery, frozen sources, semantic descriptors, and result objects have no certified physical allocation bound. A finite host-memory policy rejects before filesystem discovery or reads; this entry does not claim bounded-host completion.",
                "Zero or one output is supported. Invalid controls, discovery failures, missing executor, or unsupported infrastructure are diagnostics; discovering an invalid test body is distinct from executing a failed test.",
                "This adapter does not implement MATLAB wildcard selectors, direct file/procedure selectors, namespace/project selector semantics, or advanced parameter/dependency filters. BaseFolder supplies path-resolution roots here rather than MATLAB's wildcard base-folder filter. Unknown options fail instead of being silently accepted.",
            ],
        },
    ],
    examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does testsuite run the test bodies?", answer: "No. It returns semantic test procedure descriptors. Use runtests or a TestRunner to execute them." },
        BuiltinDocumentationFaq { question: "Does discovery read and then regenerate source?", answer: "No. Runtime reads each selected source once. Core freezes those original bytes and discovers semantic procedures without generated source calls." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "runtests", target: BuiltinDocumentationLinkTarget::Builtin("runtests") },
            BuiltinDocumentationLink { label: "MATLAB testsuite reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/testsuite.html") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Retained Runtime entry", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/testing/testsuite.rs") },
            BuiltinDocumentationLink { label: "Shared frozen-source Core discovery", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-core/src/testing/runtime_adapter.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Shared TestSuite/TestRunner discovery and selection", location: "runmat_core::tests::testsuite_and_testrunner_share_semantic_core_discovery_and_selection" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Frozen source ownership", location: "testing::runtime_adapter::tests::runtests_freezes_original_source_and_leaves_procedure_selection_to_core" },
        ],
        notes: &[
            "This is first canonical documentation for the existing public runtime identity; no legacy JSON sidecar existed. It describes the actual shared adapter rather than claiming the full MATLAB TestSuite API.",
            "Contract remains Incomplete and documentation Partial. Standalone fixtures and existing test pointers require central execution qualification; no fresh test, browser, or finite-budget success is claimed by this source migration.",
        ],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

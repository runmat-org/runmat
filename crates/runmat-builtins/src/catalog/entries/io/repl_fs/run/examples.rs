use crate::*;
pub(super) const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "current-folder",
        title: "Execute a script in the current folder",
        program: "run('setup_experiment');\nresult = configured_gain * 2;\nassert(result == 42);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId {
                local_name: "current-folder",
            },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[
                BuiltinFilesystemEntry::File {
                    relative_path: "setup_experiment.m",
                    content: BuiltinFixtureContent::Utf8("configured_gain = 21;\n"),
            }
            ],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(result == 42);",
            },
    },
    BuiltinExample {
        id: "search-path",
        title: "Execute a script found on the search path",
        program: "original = path;\naddpath('examples');\nrun fm_demo\nassert(isequal(t, [0; 1]));\nassert(isequal(demodulated, [2; 4]));\npath(original);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId {
                local_name: "search-path",
            },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[
                BuiltinFilesystemEntry::Directory {
                    relative_path: "examples",
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "examples/fm_demo.m",
                    content: BuiltinFixtureContent::Utf8("t = [0; 1]; demodulated = [2; 4];\n"),
            }
            ],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(demodulated, [2; 4]));",
            },
    },
    BuiltinExample {
        id: "catch-missing",
        title: "Catch script execution errors",
        program: "caught = false;\ntry\n    run('maybe_missing_script');\ncatch e\n    caught = strcmp(e.identifier, 'RunMat:run:FileNotFound');\nend\nassert(caught);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(caught);",
            },
    },
    BuiltinExample {
        id: "partial-assignments",
        title: "Retain assignments performed before a script error",
        program: "caught = false;\ntry\n    run('partial_script');\ncatch e\n    caught = strcmp(e.identifier, 'example:scriptFailure');\nend\nassert(caught);\nassert(partial_value == 7);\nassert(strcmp(script_context, 'partial_script'));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId {
                local_name: "partial-assignments",
            },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[
                BuiltinFilesystemEntry::File {
                    relative_path: "partial_script.m",
                    content: BuiltinFixtureContent::Utf8("partial_value = 7; script_context = mfilename; error('example:scriptFailure', 'intentional failure');\n"),
            }
            ],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(caught); assert(partial_value == 7); assert(strcmp(script_context, 'partial_script'));",
            },
    },
];

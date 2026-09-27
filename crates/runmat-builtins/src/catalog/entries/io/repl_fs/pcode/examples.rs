use crate::*;
pub(super) const EXAMPLES: &[
    BuiltinExample
] = &[
    BuiltinExample {
        id: "current",
        title: "Create P-code in the current folder",
        program: "pcode worker\nrun worker\nassert(answer == 42);\nassert(isfile('worker.p'));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId {
                local_name: "current"
            },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[
                BuiltinFilesystemEntry::File {
                    relative_path: "worker.m",
                    content: BuiltinFixtureContent::Utf8("answer = 42;\n")
                },
            ],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(answer == 42); assert(isfile('worker.p'));"
        },
    },
    BuiltinExample {
        id: "inplace",
        title: "Write P-code beside the source",
        program: "pcode src/worker.m -inplace\nassert(isfile('src/worker.p'));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId {
                local_name: "inplace"
            },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[
                BuiltinFilesystemEntry::Directory {
                    relative_path: "src"
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "src/worker.m",
                    content: BuiltinFixtureContent::Utf8("answer = 7;\n")
                },
            ],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isfile('src/worker.p'));"
        },
    },
    BuiltinExample {
        id: "multiple",
        title: "Encode multiple source patterns",
        program: "pcode src/*.m helpers/*.m -R2022a\nassert(isfile('worker.p')); assert(isfile('helper.p'));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId {
                local_name: "multiple"
            },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[
                BuiltinFilesystemEntry::Directory {
                    relative_path: "helpers"
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "helpers/helper.m",
                    content: BuiltinFixtureContent::Utf8("helper_value = 3;\n")
                },
                BuiltinFilesystemEntry::Directory {
                    relative_path: "src"
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "src/worker.m",
                    content: BuiltinFixtureContent::Utf8("answer = 7;\n")
                },
            ],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isfile('worker.p')); assert(isfile('helper.p'));"
        },
    },
];

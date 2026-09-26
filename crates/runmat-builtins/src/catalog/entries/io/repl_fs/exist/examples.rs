use crate::*;
pub(super) const EXAMPLES: &[
    BuiltinExample
] = &[
    BuiltinExample {
        id: "workspace",
        title: "Check a workspace variable",
        program: "alpha = 1;\ntypeID = exist('alpha','var');\nassert(typeID == 1);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(typeID == 1);"
        },
    },
    BuiltinExample {
        id: "folder",
        title: "Check a folder through the file filter",
        program: "typeID = exist('logs','file');\nassert(typeID == 7);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId {
                local_name: "folder"
            },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[
                BuiltinFilesystemEntry::Directory {
                    relative_path: "logs"
                },
            ],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(typeID == 7);"
        },
    },
    BuiltinExample {
        id: "builtin",
        title: "Check a builtin",
        program: "typeID = exist('sin','builtin');\nassert(typeID == 5);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(typeID == 5);"
        },
    },
    BuiltinExample {
        id: "file-kind",
        title: "Distinguish files, P-code, and wildcard names",
        program: "assert(exist('data.txt','file') == 2);\nassert(exist('packed','file') == 6);\nassert(exist('*.txt','file') == 0);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId {
                local_name: "file-kind"
            },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[
                BuiltinFilesystemEntry::File {
                    relative_path: "data.txt",
                    content: BuiltinFixtureContent::Utf8("")
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "packed.p",
                    content: BuiltinFixtureContent::Utf8("")
                },
            ],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(exist('data.txt','file') == 2); assert(exist('*.txt','file') == 0);"
        },
    },
];

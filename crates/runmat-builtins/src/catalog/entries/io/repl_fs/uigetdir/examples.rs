use crate::*;
pub(super) const EXAMPLES: &[
    BuiltinExample
] = &[
    BuiltinExample {
        id: "select",
        title: "Select a folder and list its contents",
        program: "folder = uigetdir;\nassert(ischar(folder));\nlisting = dir(folder);\nassert(~isempty(listing));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture {
            id: BuiltinExampleFixtureId {
                local_name: "select"
            },
            entries: &[
                BuiltinFilesystemEntry::Directory {
                    relative_path: "images"
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "images/input.txt",
                    content: BuiltinFixtureContent::Utf8("fixture")
                }
            ],
            interactions: &[
                BuiltinDesktopHostInteraction::DirectoryDialog(BuiltinDirectoryDialogInteraction {
                    request: BuiltinDirectoryDialogExpectation {
                        title: None,
                        default_path: None
                    },
                    outcome: BuiltinDirectoryDialogOutcome::Selection {
                        path: "images"
                    },
                })
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(folder)); assert(~isempty(listing));"
        },
    },
    BuiltinExample {
        id: "title",
        title: "Choose an initial folder and title",
        program: "folder = uigetdir('images', 'Select input image folder');\nassert(ischar(folder));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture {
            id: BuiltinExampleFixtureId {
                local_name: "title"
            },
            entries: &[
                BuiltinFilesystemEntry::Directory {
                    relative_path: "images"
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "images/input.txt",
                    content: BuiltinFixtureContent::Utf8("fixture")
                }
            ],
            interactions: &[
                BuiltinDesktopHostInteraction::DirectoryDialog(BuiltinDirectoryDialogInteraction {
                    request: BuiltinDirectoryDialogExpectation {
                        title: Some("Select input image folder"),
                        default_path: Some("images")
                    },
                    outcome: BuiltinDirectoryDialogOutcome::Selection {
                        path: "images"
                    },
                })
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(folder));"
        },
    },
    BuiltinExample {
        id: "cancel",
        title: "Handle a cancelled directory dialog",
        program: "folder = uigetdir;\nassert(isequal(folder, 0));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture {
            id: BuiltinExampleFixtureId {
                local_name: "cancel"
            },
            entries: &[
                BuiltinFilesystemEntry::Directory {
                    relative_path: "images"
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "images/input.txt",
                    content: BuiltinFixtureContent::Utf8("fixture")
                }
            ],
            interactions: &[
                BuiltinDesktopHostInteraction::DirectoryDialog(BuiltinDirectoryDialogInteraction {
                    request: BuiltinDirectoryDialogExpectation {
                        title: None,
                        default_path: None
                    },
                    outcome: BuiltinDirectoryDialogOutcome::Cancel,
                })
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(folder, 0));"
        },
    },
];

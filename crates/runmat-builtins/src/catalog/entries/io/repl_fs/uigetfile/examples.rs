use crate::*;
pub(super) const EXAMPLES: &[
    BuiltinExample
] = &[
    BuiltinExample {
        id: "read",
        title: "Select and read tabular data",
        program: "[file, path] = uigetfile('*.csv', 'Select tabular data');\nassert(strcmp(file, 'scores.csv'));\nscores = readtable(fullfile(path, file));\nassert(height(scores) == 1);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture {
            id: BuiltinExampleFixtureId {
                local_name: "read"
            },
            entries: &[
                BuiltinFilesystemEntry::File {
                    relative_path: "scores.csv",
                    content: BuiltinFixtureContent::Utf8("score\n7\n")
                },
            ],
            interactions: &[
                BuiltinDesktopHostInteraction::OpenFileDialog(BuiltinOpenFileDialogInteraction {
                    request: BuiltinOpenFileDialogExpectation {
                        title: Some("Select tabular data"),
                        default_path: None,
                        filters: &[
                            BuiltinDialogFilterExpectation {
                                patterns: &[
                                    "*.csv"
                                ],
                                description: None
                            },
                        ],
                        multiselect: false,
                    },
                    outcome: BuiltinOpenFileDialogOutcome::Selection {
                        paths: &[
                            "scores.csv"
                        ],
                        filter_index: Some(0)
                    },
                })
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(height(scores) == 1);"
        },
    },
    BuiltinExample {
        id: "filters",
        title: "Use multiple filters",
        program: "filters = {'*.m;*.mlx', 'MATLAB files'; '*.txt', 'Text files'};\n[file, path, index] = uigetfile(filters, 'Open source file');\nassert(strcmp(file, 'notes.txt')); assert(index == 2);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture {
            id: BuiltinExampleFixtureId {
                local_name: "filters"
            },
            entries: &[
                BuiltinFilesystemEntry::File {
                    relative_path: "notes.txt",
                    content: BuiltinFixtureContent::Utf8("fixture")
                },
            ],
            interactions: &[
                BuiltinDesktopHostInteraction::OpenFileDialog(BuiltinOpenFileDialogInteraction {
                    request: BuiltinOpenFileDialogExpectation {
                        title: Some("Open source file"),
                        default_path: None,
                        filters: &[
                            BuiltinDialogFilterExpectation {
                                patterns: &[
                                    "*.m",
                                    "*.mlx"
                                ],
                                description: Some("MATLAB files")
                            },
                            BuiltinDialogFilterExpectation {
                                patterns: &[
                                    "*.txt"
                                ],
                                description: Some("Text files")
                            },
                        ],
                        multiselect: false,
                    },
                    outcome: BuiltinOpenFileDialogOutcome::Selection {
                        paths: &[
                            "notes.txt"
                        ],
                        filter_index: Some(1)
                    },
                })
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(file, 'notes.txt')); assert(index == 2);"
        },
    },
    BuiltinExample {
        id: "multiple",
        title: "Select multiple files",
        program: "[files, path] = uigetfile('*.csv', 'Select CSV files', 'data', 'MultiSelect', 'on');\nassert(isequal(files, {'a.csv', 'b.csv'}));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture {
            id: BuiltinExampleFixtureId {
                local_name: "multiple"
            },
            entries: &[
                BuiltinFilesystemEntry::Directory {
                    relative_path: "data"
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "data/a.csv",
                    content: BuiltinFixtureContent::Utf8("x\n1\n")
                },
                BuiltinFilesystemEntry::File {
                    relative_path: "data/b.csv",
                    content: BuiltinFixtureContent::Utf8("x\n2\n")
                },
            ],
            interactions: &[
                BuiltinDesktopHostInteraction::OpenFileDialog(BuiltinOpenFileDialogInteraction {
                    request: BuiltinOpenFileDialogExpectation {
                        title: Some("Select CSV files"),
                        default_path: Some("data"),
                        filters: &[
                            BuiltinDialogFilterExpectation {
                                patterns: &[
                                    "*.csv"
                                ],
                                description: None
                            },
                        ],
                        multiselect: true,
                    },
                    outcome: BuiltinOpenFileDialogOutcome::Selection {
                        paths: &[
                            "data/a.csv",
                            "data/b.csv"
                        ],
                        filter_index: Some(0)
                    },
                })
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(files, {'a.csv', 'b.csv'}));"
        },
    },
    BuiltinExample {
        id: "cancel",
        title: "Handle cancellation",
        program: "[file, path, index] = uigetfile;\nassert(isequal(file, 0)); assert(isequal(path, 0)); assert(isequal(index, 0));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture {
            id: BuiltinExampleFixtureId {
                local_name: "cancel"
            },
            entries: &[
            ],
            interactions: &[
                BuiltinDesktopHostInteraction::OpenFileDialog(BuiltinOpenFileDialogInteraction {
                    request: BuiltinOpenFileDialogExpectation {
                        title: None,
                        default_path: None,
                        filters: &[
                            BuiltinDialogFilterExpectation {
                                patterns: &[
                                    "*.*"
                                ],
                                description: Some("All Files")
                            },
                        ],
                        multiselect: false,
                    },
                    outcome: BuiltinOpenFileDialogOutcome::Cancel,
                })
            ],
        }),
        requirements: BuiltinExampleRequirements {
            host: BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(file, 0)); assert(index == 0);"
        },
    },
];

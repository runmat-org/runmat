use crate::*;
pub(super) const EXAMPLES: &[
    BuiltinExample
] = &[
    BuiltinExample {
        id: "save",
        title: "Choose a MAT-file destination",
        program: "value = 42;\n[file, path] = uiputfile('*.mat', 'Save workspace');\nassert(strcmp(file, 'results.mat'));\nsave(fullfile(path, file), 'value');\nassert(isfile(fullfile(path, file)));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture {
            id: BuiltinExampleFixtureId {
                local_name: "save"
            },
            entries: &[
            ],
            interactions: &[
                BuiltinDesktopHostInteraction::SaveFileDialog(BuiltinSaveFileDialogInteraction {
                    request: BuiltinSaveFileDialogExpectation {
                        title: Some("Save workspace"),
                        default_path: None,
                        filters: &[
                            BuiltinDialogFilterExpectation {
                                patterns: &[
                                    "*.mat"
                                ],
                                description: None
                            },
                        ],
                    },
                    outcome: BuiltinSaveFileDialogOutcome::Selection {
                        path: "results.mat",
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
            source: "assert(isfile(fullfile(path, file)));"
        },
    },
    BuiltinExample {
        id: "filters",
        title: "Use multiple filters",
        program: "filters = {'*.mat', 'MAT-files'; '*.txt', 'Text files'};\n[file, path, index] = uiputfile(filters, 'Export data', 'results.mat');\nassert(strcmp(file, 'report.txt'));\nassert(index == 2);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::InteractiveHost,
        fixture: BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture {
            id: BuiltinExampleFixtureId {
                local_name: "filters"
            },
            entries: &[
            ],
            interactions: &[
                BuiltinDesktopHostInteraction::SaveFileDialog(BuiltinSaveFileDialogInteraction {
                    request: BuiltinSaveFileDialogExpectation {
                        title: Some("Export data"),
                        default_path: Some("results.mat"),
                        filters: &[
                            BuiltinDialogFilterExpectation {
                                patterns: &[
                                    "*.mat"
                                ],
                                description: Some("MAT-files")
                            },
                            BuiltinDialogFilterExpectation {
                                patterns: &[
                                    "*.txt"
                                ],
                                description: Some("Text files")
                            },
                        ],
                    },
                    outcome: BuiltinSaveFileDialogOutcome::Selection {
                        path: "report.txt",
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
            source: "assert(strcmp(file, 'report.txt')); assert(index == 2);"
        },
    },
    BuiltinExample {
        id: "cancel",
        title: "Handle cancellation",
        program: "[file, path, index] = uiputfile;\nassert(isequal(file, 0)); assert(isequal(path, 0)); assert(isequal(index, 0));",
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
                BuiltinDesktopHostInteraction::SaveFileDialog(BuiltinSaveFileDialogInteraction {
                    request: BuiltinSaveFileDialogExpectation {
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
                    },
                    outcome: BuiltinSaveFileDialogOutcome::Cancel,
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

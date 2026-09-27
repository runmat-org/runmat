use crate::*;
pub(super) const EXAMPLES:&[
    BuiltinExample
]=&[
    BuiltinExample{
        id:"line",
        title:"Open a script at a line",
        program:"opentoline('setup.m', 12)",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::InteractiveHost,
        fixture:BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture{
            id:BuiltinExampleFixtureId{
                local_name:"line"
            },
            entries:&[
                BuiltinFilesystemEntry::File{
                    relative_path:"setup.m",
                    content:BuiltinFixtureContent::Utf8("x = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\n"),
                }
            ],
            interactions:&[
                BuiltinDesktopHostInteraction::Document(BuiltinDocumentExpectation::Navigate{
                    relative_path:"setup.m",
                    line:std::num::NonZeroUsize::new(12).unwrap(),
                    column:std::num::NonZeroUsize::new(1).unwrap(),
                    selection:BuiltinDocumentSelection::Caret,
                })
            ],
        }),
        requirements:BuiltinExampleRequirements{
            host:BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification:BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample{
        id:"column",
        title:"Open a script at a line and column",
        program:"opentoline('setup.m', 12, 4)",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::InteractiveHost,
        fixture:BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture{
            id:BuiltinExampleFixtureId{
                local_name:"column"
            },
            entries:&[
                BuiltinFilesystemEntry::File{
                    relative_path:"setup.m",
                    content:BuiltinFixtureContent::Utf8("x = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\n"),
                }
            ],
            interactions:&[
                BuiltinDesktopHostInteraction::Document(BuiltinDocumentExpectation::Navigate{
                    relative_path:"setup.m",
                    line:std::num::NonZeroUsize::new(12).unwrap(),
                    column:std::num::NonZeroUsize::new(4).unwrap(),
                    selection:BuiltinDocumentSelection::Caret,
                })
            ],
        }),
        requirements:BuiltinExampleRequirements{
            host:BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification:BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample{
        id:"command",
        title:"Use command-style text numbers",
        program:"opentoline setup.m 12 4",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::InteractiveHost,
        fixture:BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture{
            id:BuiltinExampleFixtureId{
                local_name:"command"
            },
            entries:&[
                BuiltinFilesystemEntry::File{
                    relative_path:"setup.m",
                    content:BuiltinFixtureContent::Utf8("x = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\n"),
                }
            ],
            interactions:&[
                BuiltinDesktopHostInteraction::Document(BuiltinDocumentExpectation::Navigate{
                    relative_path:"setup.m",
                    line:std::num::NonZeroUsize::new(12).unwrap(),
                    column:std::num::NonZeroUsize::new(4).unwrap(),
                    selection:BuiltinDocumentSelection::Caret,
                })
            ],
        }),
        requirements:BuiltinExampleRequirements{
            host:BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification:BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample{
        id:"selection",
        title:"Request selection at a location",
        program:"opentoline('setup.m', 12, 4, 'select')",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::InteractiveHost,
        fixture:BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture{
            id:BuiltinExampleFixtureId{
                local_name:"selection"
            },
            entries:&[
                BuiltinFilesystemEntry::File{
                    relative_path:"setup.m",
                    content:BuiltinFixtureContent::Utf8("x = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\n"),
                }
            ],
            interactions:&[
                BuiltinDesktopHostInteraction::Document(BuiltinDocumentExpectation::Navigate{
                    relative_path:"setup.m",
                    line:std::num::NonZeroUsize::new(12).unwrap(),
                    column:std::num::NonZeroUsize::new(4).unwrap(),
                    selection:BuiltinDocumentSelection::Select,
                })
            ],
        }),
        requirements:BuiltinExampleRequirements{
            host:BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification:BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample{
        id:"integer",
        title:"Use typed integer positions",
        program:"opentoline('setup.m', uint64(12), uint16(4))",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::RunMat,
        harness:BuiltinExampleHarness::InteractiveHost,
        fixture:BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture{
            id:BuiltinExampleFixtureId{
                local_name:"integer"
            },
            entries:&[
                BuiltinFilesystemEntry::File{
                    relative_path:"setup.m",
                    content:BuiltinFixtureContent::Utf8("x = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\nx = 1;\n"),
                }
            ],
            interactions:&[
                BuiltinDesktopHostInteraction::Document(BuiltinDocumentExpectation::Navigate{
                    relative_path:"setup.m",
                    line:std::num::NonZeroUsize::new(12).unwrap(),
                    column:std::num::NonZeroUsize::new(4).unwrap(),
                    selection:BuiltinDocumentSelection::Caret,
                })
            ],
        }),
        requirements:BuiltinExampleRequirements{
            host:BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification:BuiltinExampleVerification::Succeeds,
    }
];

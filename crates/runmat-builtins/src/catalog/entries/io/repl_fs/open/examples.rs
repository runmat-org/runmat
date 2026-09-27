use crate::*;
pub(super) const EXAMPLES:&[
    BuiltinExample
]=&[
    BuiltinExample{
        id:"mat",
        title:"Load a MAT-file through open",
        program:"A = uint64([1, 2, 3]);\nsave('results.mat', 'A');\nS = open('results.mat');\nassert(isequal(S.A, A));\nassert(strcmp(class(S.A), 'uint64'));",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::NativeFilesystem,
        fixture:BuiltinExampleFixture::None,
        requirements:BuiltinExampleRequirements::NONE,
        verification:BuiltinExampleVerification::Assertions{
            source:"assert(isequal(S.A, A));\nassert(strcmp(class(S.A), 'uint64'));"
        },
    },
    BuiltinExample{
        id:"script",
        title:"Open a script from the current path",
        program:"open setup",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::InteractiveHost,
        fixture:BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture{
            id:BuiltinExampleFixtureId{
                local_name:"script"
            },
            entries:&[
                BuiltinFilesystemEntry::File{
                    relative_path:"setup.m",
                    content:BuiltinFixtureContent::Utf8("configured = true;\n")
                }
            ],
            interactions:&[
                BuiltinDesktopHostInteraction::Document(BuiltinDocumentExpectation::OpenFile{
                    relative_path:"setup.m"
                })
            ]
        }),
        requirements:BuiltinExampleRequirements{
            host:BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification:BuiltinExampleVerification::Succeeds,
    },
    BuiltinExample{
        id:"figure",
        title:"Open a RunMat figure scene",
        program:"original = figure;\nsavefig(original, 'plot.fig');\nclose(original);\nh = open('plot.fig');\nassert(ishandle(h));\nopened = ishandle(h);\nclose(h);",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::RunMat,
        harness:BuiltinExampleHarness::NativeFilesystem,
        fixture:BuiltinExampleFixture::None,
        requirements:BuiltinExampleRequirements::NONE,
        verification:BuiltinExampleVerification::Assertions{
            source:"assert(opened);"
        },
    },
    BuiltinExample{
        id:"handler",
        title:"Dispatch to a custom extension handler",
        program:"out = open('payload.abc');\nassert(strcmp(out, 'payload'));",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::NativeFilesystem,
        fixture:BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture{
            id:BuiltinExampleFixtureId{
                local_name:"handler"
            },
            root:BuiltinFilesystemRoot::IsolatedWorkspace,
            entries:&[
                BuiltinFilesystemEntry::File{
                    relative_path:"openabc.m",
                    content:BuiltinFixtureContent::Utf8("function out = openabc(filename)\n out = fileread(filename);\nend\n")
                },
                BuiltinFilesystemEntry::File{
                    relative_path:"payload.abc",
                    content:BuiltinFixtureContent::Utf8("payload")
                }
            ]
        }),
        requirements:BuiltinExampleRequirements::NONE,
        verification:BuiltinExampleVerification::Assertions{
            source:"assert(strcmp(out, 'payload'));"
        },
    },
    BuiltinExample{
        id:"variable",
        title:"Open a workspace variable",
        program:"results = [1, 2, 3];\nopen results",
        display_output:None,
        compatibility:BuiltinExampleCompatibility::Matlab,
        harness:BuiltinExampleHarness::InteractiveHost,
        fixture:BuiltinExampleFixture::DesktopHostOnly(BuiltinDesktopHostFixture{
            id:BuiltinExampleFixtureId{
                local_name:"variable"
            },
            entries:&[
            ],
            interactions:&[
                BuiltinDesktopHostInteraction::Document(BuiltinDocumentExpectation::OpenVariable{
                    name:"results"
                })
            ]
        }),
        requirements:BuiltinExampleRequirements{
            host:BuiltinExampleHostRequirement::DesktopHostOnly,
            ..BuiltinExampleRequirements::NONE
        },
        verification:BuiltinExampleVerification::Succeeds,
    }
];

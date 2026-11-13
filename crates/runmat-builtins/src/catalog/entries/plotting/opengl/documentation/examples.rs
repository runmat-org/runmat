use crate::*;
pub(super) const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Query renderer information",
        program: "info = opengl('info');\ninfo.Renderer",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(info.Renderer,'runmat-plot')); assert(strcmp(info.Vendor,'RunMat')); assert(~info.Software); assert(info.MaxTextureSize==16384); assert(strcmp(opengl('software'),'software'));" },
    },
];

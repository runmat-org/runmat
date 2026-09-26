use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("uint32.empty"),
    slug: Some("uint32.empty"),
    summary: "Create an empty uint32 array.",
    description: concat!(
        "The uint32 class receiver selects exact host storage. Dimension ",
        "controls determine the empty shape, not the output class.",
    ),
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Create an empty uint32 matrix",
            program: "A = uint32.empty(0, 5);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: concat!(
                    "assert(isempty(A)); assert(isequal(size(A), [0 5])); ",
                    "assert(strcmp(class(A), \"uint32\"));",
                ),
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Preserve a normalized N-D empty shape",
            program: "A = uint32.empty(int32([0 2 3 1]));",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(A), [0 2 3])); assert(strcmp(class(A), \"uint32\"));",
            },
        },
    ],
    ..super::super::documentation::DOCUMENTATION
};

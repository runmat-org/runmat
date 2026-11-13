use crate::*;
pub(super) const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "numeric-selector",
        title: "Reject numeric selector before registry access",
        program: "winqueryreg(1, 'Software');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:winqueryreg:InvalidInput",
        },
    },
    BuiltinExample {
        id: "invalid-arity",
        title: "Reject invalid arity before registry access",
        program: "winqueryreg('HKEY_CURRENT_USER');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:winqueryreg:InvalidInput",
        },
    },
];

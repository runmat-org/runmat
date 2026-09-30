use crate::{
    BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

pub(super) const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
 id: "set",
 title: "Set a variable",
 program: "setenv('RUNMAT_DOC_SETENV', 'development');\nvalue = getenv('RUNMAT_DOC_SETENV');\nunsetenv('RUNMAT_DOC_SETENV');",
 display_output: Some("value = 'development'"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(strcmp(value, 'development'));" } },

    BuiltinExample {
 id: "replace",
 title: "Replace an existing value",
 program: "setenv('RUNMAT_DOC_SETENV_REPLACE', 'first');\nsetenv('RUNMAT_DOC_SETENV_REPLACE', 'second');\nvalue = getenv('RUNMAT_DOC_SETENV_REPLACE');\nunsetenv('RUNMAT_DOC_SETENV_REPLACE');",
 display_output: Some("value = 'second'"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(strcmp(value, 'second'));" } },

    BuiltinExample {
 id: "several",
 title: "Set several variables",
 program: "setenv([\"RUNMAT_DOC_SETENV_A\", \"RUNMAT_DOC_SETENV_B\"], [\"alpha\", \"beta\"]);\nvalues = getenv([\"RUNMAT_DOC_SETENV_A\", \"RUNMAT_DOC_SETENV_B\"]);\nunsetenv([\"RUNMAT_DOC_SETENV_A\", \"RUNMAT_DOC_SETENV_B\"]);",
 display_output: Some("values = [\"alpha\" \"beta\"]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(values, [\"alpha\", \"beta\"]));" } },

    BuiltinExample {
 id: "numeric",
 title: "Set a numeric scalar value",
 program: "setenv('RUNMAT_DOC_SETENV_INTEGER', intmax('uint64'));\nvalue = getenv('RUNMAT_DOC_SETENV_INTEGER');\nunsetenv('RUNMAT_DOC_SETENV_INTEGER');",
 display_output: Some("value = '18446744073709551615'"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(strcmp(value, '18446744073709551615'));" } },

    BuiltinExample {
 id: "dictionary",
 title: "Set values from a dictionary",
 program: "environment = dictionary([\"RUNMAT_DOC_SETENV_C\", \"RUNMAT_DOC_SETENV_D\"], [\"one\", \"two\"]);\nsetenv(environment);\nvalues = getenv([\"RUNMAT_DOC_SETENV_C\", \"RUNMAT_DOC_SETENV_D\"]);\nunsetenv([\"RUNMAT_DOC_SETENV_C\", \"RUNMAT_DOC_SETENV_D\"]);",
 display_output: Some("values = [\"one\" \"two\"]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(values, [\"one\", \"two\"]));" } },

    BuiltinExample {
 id: "status-extension",
 title: "Request RunMat status outputs",
 program: "[status, message] = setenv('RUNMAT_DOC_SETENV_STATUS', 'ready');\nvalue = getenv('RUNMAT_DOC_SETENV_STATUS');\nunsetenv('RUNMAT_DOC_SETENV_STATUS');",
 display_output: Some("status = 0 and message is empty"),
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(status == 0); assert(isempty(message)); assert(strcmp(value, 'ready'));" } },

];

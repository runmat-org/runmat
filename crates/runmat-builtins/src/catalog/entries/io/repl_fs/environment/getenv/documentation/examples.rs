use crate::{
    BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

pub(super) const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
 id: "scalar",
 title: "Read one variable",
 program: "setenv('RUNMAT_DOC_GETENV_SCALAR', 'ready');\nvalue = getenv('RUNMAT_DOC_GETENV_SCALAR');\nunsetenv('RUNMAT_DOC_GETENV_SCALAR');",
 display_output: Some("value = 'ready'"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(strcmp(value, 'ready'));" } },

    BuiltinExample {
 id: "missing",
 title: "Handle an undefined variable",
 program: "unsetenv('RUNMAT_DOC_GETENV_MISSING');\nvalue = getenv('RUNMAT_DOC_GETENV_MISSING');",
 display_output: Some("value is an empty character row"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isempty(value));" } },

    BuiltinExample {
 id: "string-array",
 title: "Read several names",
 program: "setenv('RUNMAT_DOC_GETENV_A', 'alpha');\nsetenv('RUNMAT_DOC_GETENV_B', 'beta');\nvalues = getenv([\"RUNMAT_DOC_GETENV_A\", \"RUNMAT_DOC_GETENV_B\"]);\nunsetenv([\"RUNMAT_DOC_GETENV_A\", \"RUNMAT_DOC_GETENV_B\"]);",
 display_output: Some("values = [\"alpha\" \"beta\"]"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(values, [\"alpha\", \"beta\"]));" } },

    BuiltinExample {
 id: "cell",
 title: "Preserve a cell array",
 program: "setenv('RUNMAT_DOC_GETENV_C', 'first');\nsetenv('RUNMAT_DOC_GETENV_D', 'second');\nvalues = getenv({'RUNMAT_DOC_GETENV_C', 'RUNMAT_DOC_GETENV_D'});\nunsetenv({'RUNMAT_DOC_GETENV_C', 'RUNMAT_DOC_GETENV_D'});",
 display_output: Some("values is a 1-by-2 cell array of character rows"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(size(values), [1 2])); assert(strcmp(values{1}, 'first')); assert(strcmp(values{2}, 'second'));" } },

    BuiltinExample {
 id: "all",
 title: "Inspect the environment dictionary",
 program: "setenv('RUNMAT_DOC_GETENV_ALL', 'visible');\nenvironment = getenv();\nvalue = environment(\"RUNMAT_DOC_GETENV_ALL\");\nunsetenv('RUNMAT_DOC_GETENV_ALL');",
 display_output: Some("value = \"visible\""),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(value == \"visible\");" } },

    BuiltinExample {
 id: "runmat-cell-string",
 title: "Use string values inside a cell in RunMat mode",
 program: "setenv('RUNMAT_DOC_GETENV_STRING_CELL', 'value');\nvalues = getenv({\"RUNMAT_DOC_GETENV_STRING_CELL\"});\nunsetenv('RUNMAT_DOC_GETENV_STRING_CELL');",
 display_output: Some("values{1} = \"value\""),
 compatibility: BuiltinExampleCompatibility::RunMat,
 harness: BuiltinExampleHarness::Portable,
 fixture: crate::BuiltinExampleFixture::None,
 requirements: crate::BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(values{1} == \"value\");" } },

];

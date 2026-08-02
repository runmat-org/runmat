use runmat_builtins::{ResolveContext, Type, Value};

use crate::BuiltinResult;

fn testsuite_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Object {
        class_name: Some(crate::testing::TEST_SUITE_CLASS.into()),
        shape: None,
    }
}

#[runmat_macros::runtime_builtin(
    name = "testsuite",
    category = "testing",
    summary = "Discover tests and return a homogeneous TestSuite array without executing them.",
    keywords = "testsuite,testing,discovery",
    type_resolver(testsuite_type),
    builtin_path = "crate::builtins::testing::testsuite"
)]
async fn testsuite(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::testing::discover_tests(args).await
}

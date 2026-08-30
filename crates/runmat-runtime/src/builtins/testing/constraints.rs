use runmat_value::{ObjectInstance, Value};

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsEqualTo",
    category = "testing/constraints",
    summary = "Construct an equality constraint.",
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_equal_to(expected: Value) -> BuiltinResult<Value> {
    constraint(
        runmat_types::standard::UNIT_TEST_IS_EQUAL_TO,
        Some(expected),
    )
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsTrue",
    category = "testing/constraints",
    summary = "Construct a logical-true constraint.",
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_true(args: Vec<Value>) -> BuiltinResult<Value> {
    no_args("IsTrue", args)?;
    constraint(runmat_types::standard::UNIT_TEST_IS_TRUE, None)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsFalse",
    category = "testing/constraints",
    summary = "Construct a logical-false constraint.",
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_false(args: Vec<Value>) -> BuiltinResult<Value> {
    no_args("IsFalse", args)?;
    constraint(runmat_types::standard::UNIT_TEST_IS_FALSE, None)
}

fn constraint(
    class_name: runmat_types::StaticClassIdentity,
    expected: Option<Value>,
) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    let mut object = ObjectInstance::new(class_name);
    if let Some(expected) = expected {
        object
            .properties
            .insert("__runmat_expected".into(), expected);
    }
    Ok(Value::Object(object))
}

fn no_args(name: &'static str, args: Vec<Value>) -> BuiltinResult<()> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(
            crate::build_runtime_error(format!("{name}: too many input arguments"))
                .with_identifier("RunMat:Testing:InvalidConstraint")
                .with_builtin(name)
                .build(),
        )
    }
}

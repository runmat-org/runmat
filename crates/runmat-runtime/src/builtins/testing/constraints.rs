use runmat_builtins::{ObjectInstance, Value};

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsEqualTo",
    category = "testing/constraints",
    summary = "Construct an equality constraint.",
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_equal_to(expected: Value) -> BuiltinResult<Value> {
    constraint("matlab.unittest.constraints.IsEqualTo", Some(expected))
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsTrue",
    category = "testing/constraints",
    summary = "Construct a logical-true constraint.",
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_true(args: Vec<Value>) -> BuiltinResult<Value> {
    no_args("IsTrue", args)?;
    constraint("matlab.unittest.constraints.IsTrue", None)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsFalse",
    category = "testing/constraints",
    summary = "Construct a logical-false constraint.",
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_false(args: Vec<Value>) -> BuiltinResult<Value> {
    no_args("IsFalse", args)?;
    constraint("matlab.unittest.constraints.IsFalse", None)
}

fn constraint(class_name: &str, expected: Option<Value>) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    let mut object = ObjectInstance::new(class_name.into());
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

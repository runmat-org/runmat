use runmat_value::{ObjectInstance, Value};

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsEqualTo",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_equal_to(args: Vec<Value>) -> BuiltinResult<Value> {
    let [expected] = <[Value; 1]>::try_from(args).map_err(|_| {
        crate::build_runtime_error("IsEqualTo: expected exactly one input")
            .with_identifier("RunMat:Testing:InvalidConstraint")
            .with_builtin("IsEqualTo")
            .build()
    })?;
    constraint(
        runmat_types::standard::UNIT_TEST_IS_EQUAL_TO,
        Some(expected),
    )
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsTrue",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_true(args: Vec<Value>) -> BuiltinResult<Value> {
    no_args("IsTrue", args)?;
    constraint(runmat_types::standard::UNIT_TEST_IS_TRUE, None)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsFalse",
    binding_variant = "default",
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
        object.properties.insert("Expected".into(), expected);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_value_is_retained_as_the_public_property() {
        let expected = Value::Int(runmat_value::IntValue::U64(u64::MAX));
        let Value::Object(object) = is_equal_to(vec![expected.clone()]).unwrap() else {
            panic!("IsEqualTo must construct an object");
        };
        assert!(object
            .class_name
            .is(runmat_types::standard::UNIT_TEST_IS_EQUAL_TO));
        assert_eq!(object.properties.get("Expected"), Some(&expected));
        assert_eq!(object.properties.len(), 1);
    }

    #[test]
    fn boolean_constraints_reject_inputs_before_constructing_an_object() {
        for constructor in [is_true, is_false] {
            assert!(constructor(vec![]).is_ok());
            let error = constructor(vec![Value::Bool(true)]).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Testing:InvalidConstraint"));
        }
    }

    #[test]
    fn equality_constraint_rejects_missing_and_extra_inputs() {
        for args in [
            vec![],
            vec![Value::Bool(true), Value::Bool(false)],
            vec![
                Value::String("A".into()),
                Value::String("IgnoringCase".into()),
                Value::Bool(true),
            ],
        ] {
            let error = is_equal_to(args).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Testing:InvalidConstraint"));
        }
    }
}

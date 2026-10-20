use runmat_value::Value;

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsEqualTo",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_equal_to(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = runtime.as_ref();
    if args.len() != 1 {
        return Err(
            crate::build_runtime_error("IsEqualTo: expected exactly one input")
                .with_identifier("RunMat:Testing:InvalidConstraint")
                .with_builtin("IsEqualTo")
                .build(),
        );
    }
    constraint(
        runtime,
        runmat_types::standard::UNIT_TEST_IS_EQUAL_TO,
        args.first().cloned(),
    )
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsTrue",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_true(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = runtime.as_ref();
    no_args("IsTrue", args)?;
    constraint(runtime, runmat_types::standard::UNIT_TEST_IS_TRUE, None)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.constraints.IsFalse",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::constraints"
)]
fn is_false(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = runtime.as_ref();
    no_args("IsFalse", args)?;
    constraint(runtime, runmat_types::standard::UNIT_TEST_IS_FALSE, None)
}

fn constraint(
    runtime: Option<&crate::context::RuntimeContext>,
    class_name: runmat_types::StaticClassIdentity,
    expected: Option<Value>,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::testing::ensure_testing_classes();
    crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        class_name.display_name(),
        expected
            .as_ref()
            .map(|value| ("Expected", value))
            .into_iter(),
    )
}

fn no_args(name: &'static str, args: crate::call::BuiltinArguments) -> BuiltinResult<()> {
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

    fn is_equal_to(
        runtime: Option<crate::context::RuntimeContext>,
        args: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        super::is_equal_to(
            runtime,
            crate::execution::RootedValues::new(args)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        )
    }

    fn is_true(
        runtime: Option<crate::context::RuntimeContext>,
        args: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        super::is_true(
            runtime,
            crate::execution::RootedValues::new(args)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        )
    }

    fn is_false(
        runtime: Option<crate::context::RuntimeContext>,
        args: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        super::is_false(
            runtime,
            crate::execution::RootedValues::new(args)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        )
    }

    #[test]
    fn expected_value_is_retained_as_the_public_property() {
        let expected = Value::Int(runmat_value::IntValue::U64(u64::MAX));
        let result = is_equal_to(None, vec![expected.clone()]).unwrap();
        let Value::Object(object) = result.value() else {
            panic!("IsEqualTo must construct an object");
        };
        assert!(object
            .class_name()
            .is(runmat_types::standard::UNIT_TEST_IS_EQUAL_TO));
        assert_eq!(object.property("Expected"), Some(&expected));
        assert_eq!(object.property_count(), 1);
    }

    #[test]
    fn boolean_constraints_reject_inputs_before_constructing_an_object() {
        for constructor in [is_true, is_false] {
            assert!(constructor(None, vec![]).is_ok());
            let error = constructor(None, vec![Value::Bool(true)]).unwrap_err();
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
            let error = is_equal_to(None, args).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:Testing:InvalidConstraint"));
        }
    }
}

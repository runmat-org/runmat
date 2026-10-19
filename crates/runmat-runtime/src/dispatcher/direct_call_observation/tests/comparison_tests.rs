//! Contextual, exact-output witnesses for admitted comparison subroutes.
use super::*;
use runmat_value::{IntValue, StringArray};

fn binding(name: &str) -> RuntimeBuiltinBinding {
    crate::builtins::wasm_registry::register_all();
    crate::builtin::runtime_builtin_bindings_by_name(name)[0]
}

fn call(
    runtime: &RuntimeContext,
    name: &str,
    args: &[Value],
    outputs: usize,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    futures::executor::block_on(crate::call_builtin_sequence_async_with_outputs_in_context(
        runtime, name, args, outputs,
    ))
}

fn call_implicit(
    runtime: &RuntimeContext,
    name: &str,
    args: &[Value],
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    futures::executor::block_on(crate::call_builtin_sequence_async_in_context(
        runtime, name, args,
    ))
}

fn scalar_arguments(name: &str) -> Vec<Value> {
    let mut args = vec![
        Value::String("RunMat".into()),
        Value::String("runmat".into()),
    ];
    if name != "strcmpi" {
        args.push(Value::Int(IntValue::I32(3)));
    }
    args
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn three_scalar_comparisons_select_once_and_separate_output_modes() {
    for name in ["strcmpi", "strncmp", "strncmpi"] {
        let (runtime, _scoped, placement) = harness(binding(name));
        let args = scalar_arguments(name);
        let explicit = call(&runtime, name, &args, 1).expect("explicit scalar comparison");
        let expected = if name == "strncmp" {
            Value::Bool(false)
        } else {
            Value::Bool(true)
        };
        assert_eq!(
            explicit.as_slice(),
            std::slice::from_ref(&expected),
            "{name}"
        );
        let implicit = call_implicit(&runtime, name, &args).expect("implicit scalar comparison");
        assert_eq!(implicit.as_slice(), &[expected], "{name}");
        assert_eq!(placement.selections.get(), 2, "{name}");
        assert_eq!(placement.observations.get(), 2, "{name}");
        let signatures = placement.signatures.borrow();
        assert_ne!(signatures[0], signatures[1], "{name}: output mode");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prefix_count_and_immediate_control_carrier_separate_selected_facts() {
    for name in ["strncmp", "strncmpi"] {
        let (runtime, _scoped, placement) = harness(binding(name));
        let left = Value::String("RunMat".into());
        let right = Value::String(
            if name == "strncmp" {
                "Runway"
            } else {
                "runway"
            }
            .into(),
        );
        for (count, expected) in [
            (Value::Int(IntValue::I32(3)), Value::Bool(true)),
            (Value::Int(IntValue::I32(4)), Value::Bool(false)),
            (Value::Num(3.0), Value::Bool(true)),
            (Value::Int(IntValue::I32(1)), Value::Bool(true)),
            (Value::Bool(true), Value::Bool(true)),
            (Value::Int(IntValue::U16(3)), Value::Bool(true)),
        ] {
            let output = call(&runtime, name, &[left.clone(), right.clone(), count], 1)
                .expect("admitted prefix comparison");
            assert_eq!(output.as_slice(), &[expected], "{name}");
        }
        assert_eq!(placement.selections.get(), 6, "{name}");
        assert_eq!(placement.observations.get(), 6, "{name}");
        let signatures = placement.signatures.borrow();
        assert_ne!(signatures[0], signatures[1], "{name}: parsed prefix");
        assert_ne!(
            signatures[0], signatures[2],
            "{name}: typed control carrier"
        );
        assert_ne!(signatures[3], signatures[4], "{name}: logical control");
        assert_ne!(signatures[0], signatures[5], "{name}: integer dtype");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unsupported_text_carriers_and_bad_outputs_do_not_admit_comparisons() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident test carrier");
    let scalar_array = Value::StringArray(
        StringArray::new(vec!["RunMat".into()], vec![1, 1]).expect("scalar string array"),
    );
    for name in ["strcmpi", "strncmp", "strncmpi"] {
        let (runtime, _scoped, placement) = harness(binding(name));
        let base = scalar_arguments(name);
        assert!(
            call(&runtime, name, &base, 2).is_err(),
            "{name}: output preflight"
        );
        let mut numeric = base.clone();
        numeric[0] = Value::Num(5.0);
        assert_eq!(
            call(&runtime, name, &numeric, 1).unwrap().as_slice(),
            &[Value::Bool(false)],
            "{name}: direct numeric"
        );
        let mut resident = base.clone();
        resident[0] = Value::GpuTensor(allocation.handle.clone());
        assert_eq!(
            call(&runtime, name, &resident, 1).unwrap().as_slice(),
            &[Value::Bool(false)],
            "{name}: resident text"
        );
        let mut array = base.clone();
        array[0] = scalar_array.clone();
        assert!(
            call(&runtime, name, &array, 1).is_ok(),
            "{name}: array semantics"
        );
        if name != "strcmpi" {
            let mut tensor_count = base;
            tensor_count[2] =
                Value::Tensor(runmat_value::Tensor::new(vec![3.0], vec![1, 1]).unwrap());
            assert!(
                call(&runtime, name, &tensor_count, 1).is_ok(),
                "{name}: host tensor N"
            );
        }
        assert_eq!(placement.selections.get(), 0, "{name}");
        assert_eq!(placement.observations.get(), 0, "{name}");
    }
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .expect("resident input remains owned");
    assert_eq!(allocation.probe.retirements(), 0);
}

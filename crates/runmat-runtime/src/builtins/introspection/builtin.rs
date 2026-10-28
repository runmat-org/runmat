#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinOutputMode, BuiltinParamArity, BUILTIN_DESCRIPTOR,
    BUILTIN_INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Value;

use crate::execution::RootedValue;

#[cfg(test)]
#[path = "builtin/ownership_tests.rs"]
mod ownership_tests;

async fn call_registered(
    runtime: Option<&crate::context::RuntimeContext>,
    name: &str,
    arguments: crate::call::BuiltinArguments,
    requested_outputs: usize,
) -> crate::BuiltinResult<RootedValue> {
    crate::context::check_active_host()?;
    let result = crate::dispatcher::call_retained_registered_builtin_sequence_with_outputs(
        name,
        arguments,
        requested_outputs,
        runtime,
    )
    .await
    .and_then(crate::call::arguments::project_rooted_legacy_builtin_value_abi);
    crate::context::check_active_host()?;
    result
}

#[runtime_builtin(
    name = "builtin",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::builtin"
)]
pub async fn builtin_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    name: runmat_value::Utf8Text,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<RootedValue> {
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);
    call_registered(runtime.as_ref(), &name, rest, requested_outputs).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, NumericStorage, Tensor};

    fn call(name: &str, rest: Vec<Value>) -> crate::BuiltinResult<RootedValue> {
        block_on(builtin_builtin(
            None,
            name.into(),
            crate::execution::RootedValues::new(rest)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        ))
    }

    #[cfg(feature = "wgpu")]
    fn register_wgpu_provider_available() -> bool {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_ok()
            && runmat_accelerate_api::provider().is_some()
    }

    #[test]
    fn builtin_descriptor_declares_dynamic_outputs_and_integer_delegation() {
        assert_eq!(
            BUILTIN_DESCRIPTOR.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );
        assert_eq!(
            BUILTIN_DESCRIPTOR.signatures[0].outputs[0].arity,
            BuiltinParamArity::Variadic
        );
        assert_eq!(BUILTIN_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            BUILTIN_INTEGER_CAPABILITIES[0].backend,
            BuiltinIntegerBackendRule::FunctionSpecific
        );
    }

    #[test]
    fn builtin_plus_preserves_all_exact_integer_classes_and_target_saturation() {
        let cases = [
            (
                IntegerStorage::I8(vec![i8::MAX, -5]),
                IntValue::I8(1),
                IntegerStorage::I8(vec![i8::MAX, -4]),
            ),
            (
                IntegerStorage::I16(vec![i16::MAX, -5]),
                IntValue::I16(1),
                IntegerStorage::I16(vec![i16::MAX, -4]),
            ),
            (
                IntegerStorage::I32(vec![i32::MAX, -5]),
                IntValue::I32(1),
                IntegerStorage::I32(vec![i32::MAX, -4]),
            ),
            (
                IntegerStorage::I64(vec![i64::MAX, -5]),
                IntValue::I64(1),
                IntegerStorage::I64(vec![i64::MAX, -4]),
            ),
            (
                IntegerStorage::U8(vec![u8::MAX, 5]),
                IntValue::U8(1),
                IntegerStorage::U8(vec![u8::MAX, 6]),
            ),
            (
                IntegerStorage::U16(vec![u16::MAX, 5]),
                IntValue::U16(1),
                IntegerStorage::U16(vec![u16::MAX, 6]),
            ),
            (
                IntegerStorage::U32(vec![u32::MAX, 5]),
                IntValue::U32(1),
                IntegerStorage::U32(vec![u32::MAX, 6]),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
                IntValue::U64(1),
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_994]),
            ),
        ];
        for (input, scalar, expected) in cases {
            let input = Tensor::new_integer(input, vec![2, 1]).expect("input");
            let result = call("plus", vec![Value::Tensor(input), Value::Int(scalar)])
                .expect("delegated integer plus");
            let Value::Tensor(result) = result.value() else {
                panic!("expected exact integer tensor, got {result:?}");
            };
            assert_eq!(result.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn builtin_preserves_target_native_single_and_requested_outputs() {
        let left = Tensor::from_f32(vec![1.25, 2.5], vec![2, 1]).expect("left");
        let right = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("right");
        let result = call("plus", vec![Value::Tensor(left), Value::Tensor(right)])
            .expect("delegated single plus");
        let Value::Tensor(result) = result.value() else {
            panic!("expected native single tensor");
        };
        assert_eq!(
            result
                .clone()
                .into_numeric_storage()
                .expect("single storage"),
            NumericStorage::F32(vec![2.25, 3.5])
        );

        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let source = Tensor::new_integer(IntegerStorage::I16(vec![0, 7, 0, -3]), vec![2, 2])
            .expect("source");
        let result = call("find", vec![Value::Tensor(source)]).expect("delegated find");
        let Value::OutputList(outputs) = result.value() else {
            panic!("expected two requested outputs, got {result:?}");
        };
        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .all(|value| matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F64)));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn builtin_inherits_exact_resident_integer_target_behavior() {
        let _guard = crate::builtins::common::test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("provider");
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2, 1],
        )
        .expect("source");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &source)
            .expect("integer upload");
        let result = call(
            "plus",
            vec![Value::GpuTensor(handle), Value::Int(IntValue::U64(1))],
        )
        .expect("delegated resident plus");
        let Value::GpuTensor(result) = result.value() else {
            panic!("plus residency must survive builtin dispatch, got {result:?}");
        };
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(result.clone()))
                .expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_994, u64::MAX]))
        );
    }
}

//! MATLAB integer cast builtin registrations backed by shared exact storage conversion.

macro_rules! define_integer_cast_builtin {
    (
        $module:ident,
        $function:ident,
        $name:literal,
        $target:expr,
        $path:literal,
        $argument_error:path,
        $input_error:path,
        $internal_error:path
    ) => {
        pub(crate) mod $module {
            use runmat_builtins::BuiltinErrorDescriptor;
            use runmat_macros::runtime_builtin;
            use runmat_value::Value;

            use crate::builtins::common::spec::{
                BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy,
                GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
                ShapeRequirements,
            };
            use crate::builtins::math::elementwise::integer_cast::{
                cast_value, CastError, IntegerTarget,
            };
            use crate::{build_runtime_error, BuiltinResult, RuntimeError};

            #[runmat_macros::register_gpu_spec(builtin_path = $path)]
            pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
                name: $name,
                op_kind: GpuOpKind::Elementwise,
                supported_precisions: &[ScalarType::F32, ScalarType::F64],
                broadcast: BroadcastSemantics::Matlab,
                provider_hooks: &[ProviderHook::Custom("cast_to_integer")],
                constant_strategy: ConstantStrategy::InlineLiteral,
                residency: ResidencyPolicy::NewHandle,
                nan_mode: ReductionNaN::Include,
                two_pass_threshold: None,
                workgroup_size: None,
                accepts_nan_mode: false,
                notes: "Real gpuArray inputs use the provider resident integer-cast hook. Paired-complex inputs use exact owner-resolved fallback, and both return native integer gpuArray storage.",
            };

            #[runmat_macros::register_fusion_spec(builtin_path = $path)]
            pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
                name: $name,
                shape: ShapeRequirements::BroadcastCompatible,
                constant_strategy: ConstantStrategy::InlineLiteral,
                elementwise: None,
                reduction: None,
                emits_nan: false,
                notes: "Resident integer casts use provider-native integer storage; fusion can target the provider hook when integer buffers are supported.",
            };

            #[runtime_builtin(
                name = $name,
                binding_variant = "default",
                builtin_path = $path
            )]
            pub(crate) async fn $function(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
                if !rest.is_empty() {
                    return Err(error(
                        &$argument_error,
                        "too many input arguments",
                    ));
                }
                cast_value(value, $target).await.map_err(|cause| match cause {
                    CastError::Unsupported(type_name) => error(
                        &$input_error,
                        format!("conversion to {} from {type_name} is not possible", $name),
                    ),
                    CastError::Internal(detail) => error(&$internal_error, detail),
                })
            }

            fn error(descriptor: &'static BuiltinErrorDescriptor, detail: impl std::fmt::Display) -> RuntimeError {
                let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
                    .with_builtin($name);
                if let Some(identifier) = descriptor.identifier {
                    builder = builder.with_identifier(identifier);
                }
                builder.build()
            }
        }
    };
}

define_integer_cast_builtin!(
    int8,
    int8_builtin,
    "int8",
    IntegerTarget::I8,
    "crate::builtins::math::elementwise::integer_cast_builtins::int8",
    runmat_builtins::INT8_ERROR_INVALID_ARGUMENT,
    runmat_builtins::INT8_ERROR_INVALID_INPUT,
    runmat_builtins::INT8_ERROR_INTERNAL
);
define_integer_cast_builtin!(
    int16,
    int16_builtin,
    "int16",
    IntegerTarget::I16,
    "crate::builtins::math::elementwise::integer_cast_builtins::int16",
    runmat_builtins::INT16_ERROR_INVALID_ARGUMENT,
    runmat_builtins::INT16_ERROR_INVALID_INPUT,
    runmat_builtins::INT16_ERROR_INTERNAL
);
define_integer_cast_builtin!(
    int64,
    int64_builtin,
    "int64",
    IntegerTarget::I64,
    "crate::builtins::math::elementwise::integer_cast_builtins::int64",
    runmat_builtins::INT64_ERROR_INVALID_ARGUMENT,
    runmat_builtins::INT64_ERROR_INVALID_INPUT,
    runmat_builtins::INT64_ERROR_INTERNAL
);
define_integer_cast_builtin!(
    uint64,
    uint64_builtin,
    "uint64",
    IntegerTarget::U64,
    "crate::builtins::math::elementwise::integer_cast_builtins::uint64",
    runmat_builtins::UINT64_ERROR_INVALID_ARGUMENT,
    runmat_builtins::UINT64_ERROR_INVALID_INPUT,
    runmat_builtins::UINT64_ERROR_INTERNAL
);

#[cfg(test)]
mod tests {
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelProvider, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView,
        HostTensorView, IntegerElementType,
    };
    use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor, Value};

    #[test]
    fn int8_and_int16_scalars_saturate_and_round() {
        assert_eq!(
            block_on(super::int8::int8_builtin(Value::Num(127.6), Vec::new())).expect("int8"),
            Value::Int(IntValue::I8(i8::MAX))
        );
        assert_eq!(
            block_on(super::int16::int16_builtin(
                Value::Num(-32768.6),
                Vec::new()
            ))
            .expect("int16"),
            Value::Int(IntValue::I16(i16::MIN))
        );
    }

    #[test]
    fn int64_array_cast_preserves_exact_uint64_saturation() {
        let source =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("source");
        let output = block_on(super::int64::int64_builtin(
            Value::Tensor(source),
            Vec::new(),
        ))
        .expect("int64");

        assert_eq!(output, Value::Int(IntValue::I64(i64::MAX)));
    }

    #[test]
    fn uint64_array_cast_preserves_exact_signed_values() {
        let source = Tensor::new_integer(IntegerStorage::I64(vec![-1, i64::MAX]), vec![1, 2])
            .expect("source");
        let output = block_on(super::uint64::uint64_builtin(
            Value::Tensor(source),
            Vec::new(),
        ))
        .expect("uint64");

        match output {
            Value::Tensor(tensor) => assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U64(vec![0, i64::MAX as u64]))
            ),
            other => panic!("expected uint64 tensor, got {other:?}"),
        }
    }

    #[test]
    fn int16_casts_logical_and_character_arrays_with_exact_backing() {
        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical");
        let logical_output = block_on(super::int16::int16_builtin(
            Value::LogicalArray(logical),
            Vec::new(),
        ))
        .expect("int16 logical");
        let chars_output = block_on(super::int16::int16_builtin(
            Value::CharArray(runmat_value::CharArray::new_row("Az")),
            Vec::new(),
        ))
        .expect("int16 chars");

        for (output, expected) in [(logical_output, vec![1, 0]), (chars_output, vec![65, 122])] {
            match output {
                Value::Tensor(tensor) => assert_eq!(
                    tensor.integer_storage(),
                    Some(&IntegerStorage::I16(expected))
                ),
                other => panic!("expected int16 tensor, got {other:?}"),
            }
        }
    }

    #[test]
    fn uint64_gpu_input_stays_resident_with_exact_integer_storage() {
        test_support::with_test_provider(|provider| {
            let source = Tensor::new(vec![-1.0, 4.4], vec![1, 2]).expect("source");
            let handle = provider
                .upload(&HostTensorView {
                    data: &source.materialize_f64(),
                    shape: &source.shape,
                })
                .expect("upload");
            let output = block_on(super::uint64::uint64_builtin(
                Value::GpuTensor(handle),
                Vec::new(),
            ))
            .expect("uint64 GPU conversion");

            let Value::GpuTensor(handle) = output else {
                panic!("expected resident gpuArray result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            assert_eq!(
                block_on(provider.download_integer(&handle))
                    .expect("download uint64 cast")
                    .data,
                HostIntegerDataOwned::U64(vec![0, 4])
            );
        });
    }

    #[test]
    fn integer_cast_gpu_dispatch_uses_input_handle_owner() {
        let _guard = test_support::accel_test_lock();
        let provider_a: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        let provider_b: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        unsafe {
            runmat_accelerate_api::register_provider(provider_a);
            runmat_accelerate_api::register_provider(provider_b);
        }
        let _current = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider_b));

        let input = provider_a
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[0, 1_u64 << 63, u64::MAX]),
                shape: &[1, 3],
            })
            .expect("upload provider-a integer");
        assert_eq!(input.device_id, provider_a.device_id());
        assert_eq!(
            runmat_accelerate_api::provider()
                .expect("current provider")
                .device_id(),
            provider_b.device_id()
        );

        let output = block_on(super::int64::int64_builtin(
            Value::GpuTensor(input),
            Vec::new(),
        ))
        .expect("int64 GPU conversion");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident gpuArray output");
        };
        assert_eq!(output.device_id, provider_a.device_id());
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(IntegerElementType::I64)
        );
        assert_eq!(
            block_on(provider_a.download_integer(&output))
                .expect("download provider-a cast")
                .data,
            HostIntegerDataOwned::I64(vec![0, i64::MAX, i64::MAX])
        );
    }

    #[test]
    fn integer_casts_preserve_complex_storage_even_when_imaginary_part_rounds_to_zero() {
        let output = block_on(super::uint64::uint64_builtin(
            Value::Complex(1.0, 1e-48),
            Vec::new(),
        ))
        .expect("complex input should convert");
        assert!(matches!(
            output,
            Value::ComplexTensor(tensor)
                if tensor.integer_storage().as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![1]),
                        &IntegerStorage::U64(vec![0]),
                    ))
        ));
    }
}

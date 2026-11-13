//! Four-quadrant inverse tangent in degrees, with numeric owner preservation.
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinErrorDescriptor, ATAN2D_CHARACTER_INPUT_EXTENSION, ATAN2D_ERROR_COMPLEX_UNSUPPORTED,
    ATAN2D_ERROR_INTERNAL, ATAN2D_ERROR_INVALID_INPUT, ATAN2D_ERROR_SIZE_MISMATCH,
    ATAN2D_ERROR_TOO_MANY_OUTPUTS, ATAN2D_INTEGER_INPUT_EXTENSION, ATAN2D_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::{
    binary as common_binary, broadcast::BroadcastPlan, gpu_helpers, tensor,
};
use crate::execution::{RootedValue, RootedValueLease};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "atan2d";
mod tabular;
mod admission;

#[runtime_builtin(
    name = "atan2d",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::atan2d"
)]
async fn atan2d_builtin(y: Value, x: Value) -> BuiltinResult<RootedValue> {
    let _inputs = RootedValueLease::for_values([&y, &x].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::ATAN2D_DESCRIPTOR,
        &ATAN2D_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )?;
    match crate::builtins::table::plan_binary(y, x)
        .map_err(|cause| error(&ATAN2D_ERROR_INVALID_INPUT, cause))?
    {
        common_binary::BinaryInputPlan::Values(values) => {
            let (y, x) = *values;
            admission::pair(&y, &x)?;
            admission::host_integer_exactness(&y, &x)?;
            admission::exact_integer_boundaries(&y, &x).await?;
            let output = atan2d_admitted(y, x, &[]).await?;
            RootedValue::new(output).map_err(|cause| cause.into_runtime_error())
        }
        common_binary::BinaryInputPlan::Structured(plan) => {
            let crate::builtins::table::TabularBinaryPlan {
                source,
                variables,
                right_values,
            } = plan;
            tabular::execute(source, variables, right_values).await
        }
    }
}

async fn atan2d_admitted(
    y: Value,
    x: Value,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let mut sources = Vec::with_capacity(2);
    for input in [&y, &x] {
        if let Value::GpuTensor(handle) = input {
            if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real {
                return Err(error(&ATAN2D_ERROR_COMPLEX_UNSUPPORTED, "complex input"));
            }
            sources.push(handle.clone());
        }
    }
    let owner: Option<&'static dyn AccelProvider> = match sources.as_slice() {
        [] => None,
        [one] => Some(
            gpu_helpers::exact_provider_for_handle(one)
                .ok_or_else(|| error(&ATAN2D_ERROR_INTERNAL, "GPU input has no owning provider"))?,
        ),
        [left, right] => Some(
            gpu_helpers::exact_provider_for_binary_inputs(left, right)
                .map_err(|cause| error(&ATAN2D_ERROR_INVALID_INPUT, cause))?,
        ),
        _ => unreachable!("two input operands"),
    };
    let y = gather_if_resident(y).await?;
    let x = gather_if_resident(x).await?;
    let output = host(y, x)?;
    let Some(owner) = owner else {
        return Ok(output);
    };
    let prototype = sources
        .iter()
        .find(|source| runmat_accelerate_api::handle_is_explicit(source))
        .unwrap_or(&sources[0]);
    let mut protected_handles = Vec::with_capacity(sources.len() + protected.len());
    protected_handles.extend_from_slice(&sources);
    protected_handles.extend_from_slice(protected);
    crate::builtins::common::provider_restore::upload_value_like_protected(
        owner,
        output,
        NAME,
        prototype,
        &protected_handles,
    )
    .map_err(|cause| contextual_error(&ATAN2D_ERROR_INTERNAL, cause))
}

async fn gather_if_resident(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|cause| contextual_error(&ATAN2D_ERROR_INTERNAL, cause)),
        other => Ok(other),
    }
}

fn host(y: Value, x: Value) -> BuiltinResult<Value> {
    let y = input_tensor(y)?;
    let x = input_tensor(x)?;
    let result =
        super::atan2::binary_angle::compute(&y, &x, super::atan2::binary_angle::Unit::Degrees)
            .map_err(|cause| match cause {
                super::atan2::binary_angle::ComputeError::SizeMismatch(detail) => {
                    error(&ATAN2D_ERROR_SIZE_MISMATCH, detail)
                }
                super::atan2::binary_angle::ComputeError::Internal(detail) => {
                    error(&ATAN2D_ERROR_INTERNAL, detail)
                }
            })?;
    Ok(tensor::tensor_into_value(result))
}

fn input_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(error(&ATAN2D_ERROR_COMPLEX_UNSUPPORTED, "complex input"))
        }
        Value::CharArray(chars) => Tensor::new(
            chars
                .data()
                .iter()
                .map(|character| u32::from(*character) as f64)
                .collect(),
            vec![chars.rows, chars.cols],
        )
        .map_err(|cause| error(&ATAN2D_ERROR_INTERNAL, cause)),
        other => tensor::value_into_tensor_for(NAME, other)
            .map_err(|cause| error(&ATAN2D_ERROR_INVALID_INPUT, cause)),
    }
}

fn error(
    descriptor: &'static BuiltinErrorDescriptor,
    cause: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {cause}", descriptor.message)).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn contextual_error(
    descriptor: &'static BuiltinErrorDescriptor,
    cause: RuntimeError,
) -> RuntimeError {
    let mut contextual = error(descriptor, cause.message());
    contextual.source = Some(Box::new(cause));
    contextual
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::trigonometry::test_owner;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelNumericDownloadFuture, HostTensorView, ThreadProviderGuard,
    };

    struct NoReadProvider(test_owner::Owner);
    impl AccelProvider for NoReadProvider {
        fn device_id(&self) -> u32 {
            self.0.device_id()
        }
        fn device_info(&self) -> String {
            "atan2d preeffect fixture".into()
        }
        fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
            self.0.upload(host)
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { panic!("invalid atan2d input must not download") })
        }
        fn download_numeric<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> AccelNumericDownloadFuture<'a> {
            Box::pin(async { panic!("invalid atan2d input must not download numeric storage") })
        }
    }

    #[test]
    fn four_quadrants_and_signed_zero_use_degrees() {
        let y = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0, -1.0], vec![1, 4]).unwrap());
        let x = Value::Tensor(Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap());
        let actual = futures::executor::block_on(atan2d_builtin(y, x)).unwrap();
        let Value::Tensor(actual) = actual.value() else {
            panic!("four-angle array")
        };
        assert_eq!(actual.materialize_f64(), vec![0.0, 90.0, 180.0, -90.0]);
        let broadcast = futures::executor::block_on(atan2d_builtin(
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
        ))
        .unwrap();
        let Value::Tensor(broadcast) = broadcast.value() else {
            panic!("scalar expansion result")
        };
        assert_eq!(broadcast.materialize_f64(), vec![0.0, 45.0]);
        let both_single = futures::executor::block_on(atan2d_builtin(
            Value::Tensor(Tensor::from_f32(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::from_f32(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        ))
        .unwrap();
        let Value::Tensor(both_single) = both_single.value() else {
            panic!("single angle array")
        };
        assert_eq!(both_single.as_f32_slice(), Some([0.0, 90.0].as_slice()));
        let mixed = futures::executor::block_on(atan2d_builtin(
            Value::Tensor(Tensor::from_f32(vec![1.4e32_f32, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![-5.305e32, 1.0], vec![1, 2]).unwrap()),
        ))
        .unwrap();
        let Value::Tensor(mixed) = mixed.value() else {
            panic!("mixed precision angle array")
        };
        let expected = super::super::atan2::binary_angle::Unit::Degrees
            .f64(1.4e32_f32 as f64, -5.305e32) as f32;
        assert_eq!(mixed.as_f32_slice(), Some([expected, 0.0].as_slice()));
        assert_eq!(
            super::super::atan2::binary_angle::Unit::Degrees
                .f64(-0.0, -0.0)
                .to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn complex_coordinates_are_rejected() {
        let error =
            futures::executor::block_on(atan2d_builtin(Value::Complex(1.0, 1.0), Value::Num(1.0)))
                .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:atan2d:ComplexUnsupported"));
    }

    #[test]
    fn bad_direct_or_later_tabular_column_rejects_before_provider_read() {
        let _serial = crate::builtins::common::test_support::accel_test_lock();
        let provider = Box::leak(Box::new(NoReadProvider(test_owner::Owner::new())));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = Value::GpuTensor(
            provider
                .0
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[2, 1],
                })
                .unwrap(),
        );
        let error =
            futures::executor::block_on(atan2d_builtin(source.clone(), Value::Complex(1.0, 1.0)))
                .unwrap_err();
        assert_eq!(
            error.identifier(),
            ATAN2D_ERROR_COMPLEX_UNSUPPORTED.identifier
        );
        let table = crate::builtins::table::table_from_columns(
            vec!["valid".into(), "invalid".into()],
            vec![
                source.clone(),
                Value::ComplexTensor(
                    runmat_value::ComplexTensor::new(vec![(1.0, 1.0); 2], vec![2, 1]).unwrap(),
                ),
            ],
        )
        .unwrap();
        let error =
            futures::executor::block_on(atan2d_builtin(table, Value::Num(1.0))).unwrap_err();
        assert_eq!(
            error.identifier(),
            ATAN2D_ERROR_COMPLEX_UNSUPPORTED.identifier
        );
        assert!(
            matches!(source, Value::GpuTensor(handle) if handle.allocation_lease().validate_live().is_ok())
        );
    }

    #[test]
    fn tabular_columns_preserve_names_and_publish_degree_angles() {
        let table = crate::builtins::table::table_from_columns(
            vec!["North".into(), "West".into()],
            vec![Value::Num(1.0), Value::Num(0.0)],
        )
        .unwrap();
        let output = futures::executor::block_on(atan2d_builtin(table, Value::Num(1.0))).unwrap();
        let Value::Object(object) = output.value() else {
            panic!("tabular result")
        };
        let variables = crate::builtins::table::table_variables(object).unwrap();
        assert!(
            matches!(&variables.fields["North"], Value::Num(angle) if (angle - 45.0).abs() < 1e-12)
        );
        assert!(matches!(&variables.fields["West"], Value::Num(angle) if *angle == 0.0));
    }
}

//! Owning zero/one construction from an immutable resident prototype.
use crate::builtins::common::{
    gpu_helpers, random_args::validate_constructor_gpu_output_protecting,
};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostLogicalTensorView, HostNumericDataOwned,
    HostNumericTensorOwned, NumericElementType,
};
use runmat_value::Value;
#[path = "resident_constant/floating.rs"]
mod floating;
pub(super) use floating::floating_constant;

#[derive(Clone, Copy)]
pub(super) enum Constant {
    Zero,
    One,
}

impl Constant {
    fn bit(self) -> bool {
        matches!(self, Self::One)
    }
    fn name(self) -> &'static str {
        if self.bit() {
            "ones"
        } else {
            "zeros"
        }
    }
}

fn count(shape: &[usize]) -> BuiltinResult<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| "constant constructor: dimensions overflow".into())
}

fn provider_failure(
    name: &str,
    detail: &str,
    cause: runmat_accelerate_api::ProviderExecutionError,
) -> crate::RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        build_runtime_error(format!("{name}: {detail}: {cause}"))
            .with_builtin(name)
            .build(),
        cause.with_unproven_effects(),
    )
}

pub(super) fn logical(
    provider: &'static dyn AccelProvider,
    shape: &[usize],
    one: bool,
    protected: &[&GpuTensorHandle],
    name: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let len = count(shape)?;
    let output = match provider.logical_fill(shape, one) {
        Ok(Some(output)) => output,
        Ok(None) => {
            let data = vec![u8::from(one); len];
            runmat_accelerate_api::upload_logical_protecting(
                provider,
                &HostLogicalTensorView { data: &data, shape },
                protected,
            )
            .map_err(|cause| {
                build_runtime_error(format!("{name}: logical upload failed: {cause}"))
                    .with_builtin(name)
                    .with_source(cause)
                    .build()
            })?
        }
        Err(cause) => return Err(provider_failure(name, "logical allocation failed", cause)),
    };
    validate_constructor_gpu_output_protecting(
        name,
        provider,
        output,
        shape,
        GpuTensorStorage::Real,
        None,
        None,
        true,
        protected,
    )
}

pub(super) fn like(
    prototype: &GpuTensorHandle,
    shape: &[usize],
    constant: Constant,
) -> BuiltinResult<Value> {
    let name = constant.name();
    let len = count(shape)?;
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        build_runtime_error(format!("{name}: no provider owns the prototype"))
            .with_builtin(name)
            .build()
    })?;
    let logical_input = prototype.view().is_logical();
    let storage = prototype.physical().storage();
    let precision = runmat_accelerate_api::handle_precision(prototype);
    let integer = runmat_accelerate_api::handle_integer_type(prototype);
    let output = if logical_input {
        logical(provider, shape, constant.bit(), &[prototype], name)?
    } else {
        let attempt = match constant {
            Constant::Zero => Some(provider.zeros_like(prototype, shape)),
            Constant::One if prototype.shape() == shape => Some(provider.ones_like(prototype)),
            Constant::One
                if storage == GpuTensorStorage::Real
                    && integer.is_none()
                    && precision == Some(provider.precision()) =>
            {
                Some(provider.ones(shape))
            }
            Constant::One => None,
        };
        let direct = match attempt {
            Some(Ok(Some(output))) => Some(output),
            Some(Ok(None)) => {
                if constant.bit()
                    && storage == GpuTensorStorage::Real
                    && integer.is_none()
                    && precision == Some(provider.precision())
                {
                    match provider.fill(shape, 1.0) {
                        Ok(output) => output,
                        Err(cause) => {
                            return Err(provider_failure(name, "provider fill failed", cause))
                        }
                    }
                } else {
                    None
                }
            }
            Some(Err(cause)) => {
                return Err(provider_failure(name, "provider allocation failed", cause))
            }
            None => None,
        };
        if let Some(output) = direct {
            output
        } else {
            let host = numeric_constant(
                prototype.physical().element_type(),
                storage,
                shape,
                len,
                constant,
            )?;
            runmat_accelerate_api::upload_numeric_protecting(
                provider,
                &host.as_view(),
                &[prototype],
            )
            .map_err(|cause| {
                build_runtime_error(format!("{name}: numeric upload failed: {cause}"))
                    .with_builtin(name)
                    .with_source(cause)
                    .build()
            })?
        }
    };
    let output = validate_constructor_gpu_output_protecting(
        name,
        provider,
        output,
        shape,
        storage,
        if logical_input { None } else { precision },
        if logical_input { None } else { integer },
        logical_input,
        &[prototype],
    )?;
    let output = gpu_helpers::propagate_owned_output_provenance(output, [prototype])?;
    gpu_helpers::resident_gpu_value(output)
}

fn numeric_constant(
    element: NumericElementType,
    storage: GpuTensorStorage,
    shape: &[usize],
    elements: usize,
    constant: Constant,
) -> BuiltinResult<HostNumericTensorOwned> {
    let lanes = if storage == GpuTensorStorage::ComplexInterleaved {
        2
    } else {
        1
    };
    let count = elements
        .checked_mul(lanes)
        .ok_or("constant constructor: component count overflow")?;
    macro_rules! data {
        ($variant:ident, $ty:ty) => {{
            let mut values = vec![0 as $ty; count];
            if constant.bit() {
                for value in values.iter_mut().step_by(lanes) {
                    *value = 1 as $ty;
                }
            }
            HostNumericDataOwned::$variant(values)
        }};
    }
    let data = match element {
        NumericElementType::F64 => data!(F64, f64),
        NumericElementType::F32 => data!(F32, f32),
        NumericElementType::I8 => data!(I8, i8),
        NumericElementType::I16 => data!(I16, i16),
        NumericElementType::I32 => data!(I32, i32),
        NumericElementType::I64 => data!(I64, i64),
        NumericElementType::U8 => data!(U8, u8),
        NumericElementType::U16 => data!(U16, u16),
        NumericElementType::U32 => data!(U32, u32),
        NumericElementType::U64 => data!(U64, u64),
    };
    Ok(HostNumericTensorOwned {
        data,
        shape: shape.to_vec(),
        storage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    #[test]
    fn device_constants_preserve_every_numeric_component_class() {
        test_support::with_test_provider(|provider| {
            for element in [
                NumericElementType::F32,
                NumericElementType::F64,
                NumericElementType::I8,
                NumericElementType::I16,
                NumericElementType::I32,
                NumericElementType::I64,
                NumericElementType::U8,
                NumericElementType::U16,
                NumericElementType::U32,
                NumericElementType::U64,
            ] {
                for storage in [GpuTensorStorage::Real, GpuTensorStorage::ComplexInterleaved] {
                    let host =
                        numeric_constant(element, storage, &[1, 1], 1, Constant::One).unwrap();
                    let input = provider.upload_numeric(&host.as_view()).unwrap();
                    for constant in [Constant::Zero, Constant::One] {
                        let Value::GpuTensor(output) = like(&input, &[2, 3], constant).unwrap()
                        else {
                            panic!("resident constant")
                        };
                        assert!(!output.shares_allocation_with(&input));
                        assert_eq!(output.physical().element_type(), element);
                        assert_eq!(output.physical().storage(), storage);
                        let actual = block_on(provider.download_numeric(&output)).unwrap();
                        let expected =
                            numeric_constant(element, storage, &[2, 3], 6, constant).unwrap();
                        assert_eq!(actual, expected);
                    }
                    assert_eq!(block_on(provider.download_numeric(&input)).unwrap(), host);
                }
            }
        });
    }

    #[test]
    fn logical_constants_have_logical_producer_and_preserve_input() {
        test_support::with_test_provider(|provider| {
            let input = provider
                .upload_logical(&HostLogicalTensorView {
                    data: &[1],
                    shape: &[1, 1],
                })
                .unwrap();
            for constant in [Constant::Zero, Constant::One] {
                let output = like(&input, &[2, 3], constant).unwrap();
                let Value::GpuTensor(handle) = &output else {
                    panic!("resident logical")
                };
                assert!(handle.view().is_logical());
                assert!(!handle.shares_allocation_with(&input));
                let host = test_support::gather(output).unwrap();
                assert_eq!(
                    host.materialize_f64(),
                    vec![f64::from(u8::from(constant.bit())); 6]
                );
            }
            assert_eq!(
                test_support::gather(Value::GpuTensor(input))
                    .unwrap()
                    .materialize_f64(),
                vec![1.0]
            );
        });
    }

    #[test]
    fn zero_extents_precede_overflow_and_complex_components_are_checked() {
        assert_eq!(count(&[usize::MAX, 2, 0]).unwrap(), 0);
        assert!(count(&[usize::MAX, 2]).is_err());
        assert!(numeric_constant(
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
            &[usize::MAX],
            usize::MAX,
            Constant::One
        )
        .is_err());
    }
}

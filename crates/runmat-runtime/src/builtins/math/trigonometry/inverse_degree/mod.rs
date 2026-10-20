//! Degree-valued principal inverse trigonometric functions.
//!
//! These routes deliberately use the numerical host owner. Resident inputs
//! gather and publish through the original provider; no degree-specific
//! provider hook or profitable-placement estimate is claimed.
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::{random_args::complex_tensor_into_value, tensor};
use crate::execution::RootedValueLease;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod acosd;
mod acotd;
mod acscd;
mod asecd;
mod asind;
mod atand;

#[cfg(target_arch = "wasm32")]
pub(crate) use acosd::__runmat_wasm_register_builtin_acosd_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use acotd::__runmat_wasm_register_builtin_acotd_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use acscd::__runmat_wasm_register_builtin_acscd_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use asecd::__runmat_wasm_register_builtin_asecd_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use asind::__runmat_wasm_register_builtin_asind_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use atand::__runmat_wasm_register_builtin_atand_builtin;

const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

#[derive(Clone, Copy)]
pub(super) enum Function {
    Asin,
    Acos,
    Atan,
    Asec,
    Acsc,
    Acot,
}

pub(super) struct Spec {
    name: &'static str,
    function: Function,
    invalid_input: &'static BuiltinErrorDescriptor,
    internal: &'static BuiltinErrorDescriptor,
    integer: &'static BuiltinExtensionDescriptor,
    logical: &'static BuiltinExtensionDescriptor,
    character: &'static BuiltinExtensionDescriptor,
    gpu_complex: &'static BuiltinExtensionDescriptor,
}

impl Function {
    fn evaluate_real(self, input: f64) -> Complex64 {
        if input.is_nan() {
            return Complex64::new(f64::NAN, 0.0);
        }
        if matches!(self, Self::Acos | Self::Asec) {
            let cosine = if matches!(self, Self::Asec) {
                input.recip()
            } else {
                input
            };
            let (real, imaginary) = if (-1.0..=1.0).contains(&cosine) {
                (cosine.acos(), 0.0)
            } else if cosine > 1.0 {
                (0.0, -cosine.acosh())
            } else {
                (std::f64::consts::PI, -(-cosine).acosh())
            };
            return Complex64::new(real * RAD_TO_DEG, imaginary * RAD_TO_DEG);
        }
        if matches!(self, Self::Acsc) && input.abs() < 1.0 {
            return Complex64::new(input.recip(), 0.0).asin() * RAD_TO_DEG;
        }
        let real_radians = match self {
            Self::Asin if (-1.0..=1.0).contains(&input) => Some(input.asin()),
            Self::Atan => Some(input.atan()),
            Self::Acsc if input.abs() >= 1.0 => Some(input.recip().asin()),
            Self::Acot if input == 0.0 => Some(std::f64::consts::FRAC_PI_2),
            Self::Acot => Some(input.recip().atan()),
            _ => None,
        };
        real_radians.map_or_else(
            || self.evaluate(Complex64::new(input, 0.0)),
            |radians| Complex64::new(radians * RAD_TO_DEG, 0.0),
        )
    }

    fn evaluate(self, input: Complex64) -> Complex64 {
        let radians = match self {
            Self::Asin => input.asin(),
            Self::Acos => input.acos(),
            Self::Atan => input.atan(),
            Self::Asec => (Complex64::new(1.0, 0.0) / input).acos(),
            Self::Acsc => (Complex64::new(1.0, 0.0) / input).asin(),
            Self::Acot if input == Complex64::new(0.0, 0.0) => {
                Complex64::new(std::f64::consts::FRAC_PI_2, 0.0)
            }
            Self::Acot => (Complex64::new(1.0, 0.0) / input).atan(),
        };
        radians * RAD_TO_DEG
    }
}

pub(super) async fn execute(spec: &'static Spec, value: Value) -> BuiltinResult<Value> {
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    super::inverse_helpers::reject_excess_outputs(spec.name)?;
    super::inverse_helpers::ensure_input_extensions(
        &value,
        spec.name,
        spec.integer,
        spec.logical,
        spec.character,
    )?;
    crate::builtins::common::validation::ensure_runmat_real_integer_f64_boundary(
        &value,
        spec.integer,
        runmat_builtins::BuiltinCatalogIdentity { name: spec.name },
        spec.invalid_input,
        "X",
    )
    .await?;
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(error(
            spec,
            spec.invalid_input,
            "complex integer input is unsupported",
        ));
    }
    match value {
        Value::GpuTensor(handle) => gpu(spec, handle).await,
        other => host(spec, other),
    }
}

async fn gpu(spec: &'static Spec, handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return super::inverse_helpers::resident_complex::compute(handle, spec.name, |value| {
            host(spec, value)
        })
        .await;
    }
    super::inverse_helpers::real_fallback::compute(handle, spec.name, spec.gpu_complex, |tensor| {
        map_real(spec, tensor)
    })
    .await
}

fn host(spec: &Spec, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(re, im) => {
            let result = spec.function.evaluate(Complex64::new(re, im));
            Ok(Value::Complex(result.re, result.im))
        }
        Value::ComplexTensor(tensor) => map_complex(spec, tensor),
        Value::CharArray(chars) => {
            let tensor = Tensor::new(
                chars
                    .data()
                    .iter()
                    .map(|character| u32::from(*character) as f64)
                    .collect(),
                vec![chars.rows, chars.cols],
            )
            .map_err(|detail| error(spec, spec.invalid_input, detail))?;
            map_real(spec, tensor)
        }
        Value::String(_) | Value::StringArray(_) => {
            Err(error(spec, spec.invalid_input, "expected numeric input"))
        }
        other => {
            let tensor = tensor::value_into_tensor_for(spec.name, other)
                .map_err(|detail| error(spec, spec.invalid_input, detail))?;
            map_real(spec, tensor)
        }
    }
}

fn map_real(spec: &Spec, tensor: Tensor) -> BuiltinResult<Value> {
    super::inverse_helpers::map_real_tensor_promoting(
        tensor,
        spec.name,
        |value| {
            let result = spec.function.evaluate_real(value);
            (result.re, result.im)
        },
        |value| {
            let result = spec.function.evaluate_real(f64::from(value));
            (result.re as f32, result.im as f32)
        },
    )
    .map_err(|cause| contextual_error(spec, spec.internal, cause))
}

fn map_complex(spec: &Spec, tensor: ComplexTensor) -> BuiltinResult<Value> {
    super::inverse_helpers::map_complex_tensor(
        tensor,
        spec.name,
        |(re, im)| {
            let result = spec.function.evaluate(Complex64::new(re, im));
            (result.re, result.im)
        },
        |(re, im)| {
            let result = spec
                .function
                .evaluate(Complex64::new(f64::from(re), f64::from(im)));
            (result.re as f32, result.im as f32)
        },
    )
    .map(complex_tensor_into_value)
    .map_err(|cause| contextual_error(spec, spec.internal, cause))
}

fn contextual_error(
    spec: &Spec,
    descriptor: &'static BuiltinErrorDescriptor,
    cause: RuntimeError,
) -> RuntimeError {
    let mut contextual = error(spec, descriptor, cause.message());
    contextual.source = Some(Box::new(cause));
    contextual
}

fn error(
    spec: &Spec,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(spec.name);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::NumericDType;

    #[test]
    fn canonical_scalars_and_complex_promotion() {
        for (spec, input, expected) in [
            (&asind::SPEC, 1.0, 90.0),
            (&acosd::SPEC, 0.0, 90.0),
            (&atand::SPEC, 1.0, 45.0),
            (&asecd::SPEC, 1.0, 0.0),
            (&acscd::SPEC, 1.0, 90.0),
            (&acotd::SPEC, 0.0, 90.0),
        ] {
            let result = futures::executor::block_on(execute(spec, Value::Num(input))).unwrap();
            let Value::Num(actual) = result else {
                panic!("{} returned a non-real scalar", spec.name)
            };
            assert!((actual - expected).abs() < 1e-12, "{}", spec.name);
        }
        for (spec, input) in [
            (&asind::SPEC, 2.0),
            (&acosd::SPEC, 2.0),
            (&asecd::SPEC, 0.5),
            (&acscd::SPEC, 0.5),
        ] {
            let result = futures::executor::block_on(execute(spec, Value::Num(input))).unwrap();
            assert!(
                matches!(result, Value::Complex(_, _)),
                "{} must promote",
                spec.name
            );
        }
    }

    #[test]
    fn single_and_complex_array_shapes_are_preserved() {
        let input = Value::Tensor(Tensor::from_f32(vec![-1.0, 0.0, 1.0], vec![1, 3]).unwrap());
        let result = futures::executor::block_on(execute(&atand::SPEC, input)).unwrap();
        let Value::Tensor(result) = result else {
            panic!("single real array")
        };
        assert_eq!(result.shape(), &[1, 3]);
        assert_eq!(result.numeric_dtype(), NumericDType::F32);

        let input = Value::ComplexTensor(
            ComplexTensor::from_f32(vec![(1.0, 1.0), (0.0, -1.0)], vec![1, 2]).unwrap(),
        );
        let result = futures::executor::block_on(execute(&acotd::SPEC, input)).unwrap();
        let Value::ComplexTensor(result) = result else {
            panic!("single complex array")
        };
        assert_eq!(result.shape(), &[1, 2]);
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
    }
}

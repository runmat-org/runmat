//! The `interp2` view of the shared tensor-product not-a-knot spline.

use super::super::pp::{AdmittedSplineWork, SplineCooperation, TensorProductSpline2D};
use super::*;
use crate::context::host_allocation::HostAllocationReceipt;
use runmat_value::{ExactHostComplexTensorAllocation, ExactHostTensorAllocation};

pub(super) async fn evaluate(
    parsed: &ParsedInterp2,
    runtime: Option<&crate::context::RuntimeContext>,
    receipt: Option<&HostAllocationReceipt>,
) -> crate::BuiltinResult<Value> {
    let count = parsed.xq.len();
    let bytes_per_sample = if parsed.z.is_complex() {
        16usize
    } else {
        8usize
    };
    let (rows, cols) = parsed.z.dimensions();
    let grid_count = rows
        .checked_mul(cols)
        .ok_or_else(|| interp2_invalid_input("spline grid size overflow"))?;
    let lanes = if parsed.z.is_complex() {
        2usize
    } else {
        1usize
    };
    grid_count
        .checked_mul(3)
        .and_then(|size| size.checked_mul(lanes))
        .and_then(|size| size.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| interp2_invalid_input("spline derivative workspace size overflow"))?;
    count
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| interp2_invalid_input("spline output size overflow"))?;
    let mut work = AdmittedSplineWork::new(runtime, receipt, NAME);
    let allocation = receipt.zip(runtime);
    let real = TensorProductSpline2D::prepare(
        &parsed.x_axis,
        &parsed.y_axis,
        |index| parsed.z.at(index, false),
        &mut work,
        NAME,
        allocation,
    )
    .await
    .map_err(|err| interp2_map_error(err, &INTERP2_ERROR_INTERNAL))?;
    let imaginary = if parsed.z.is_complex() {
        Some(
            TensorProductSpline2D::prepare(
                &parsed.x_axis,
                &parsed.y_axis,
                |index| parsed.z.at(index, true),
                &mut work,
                NAME,
                allocation,
            )
            .await
            .map_err(|err| interp2_map_error(err, &INTERP2_ERROR_INTERNAL))?,
        )
    } else {
        None
    };

    let mut output = SplineOutput::new(parsed, receipt, runtime)?;
    for (index, (&xq, &yq)) in parsed.xq.iter().zip(&parsed.yq).enumerate() {
        if index.is_multiple_of(4096) {
            work.checkpoint(if index == 0 { 0 } else { 4096 })
                .await
                .map_err(|err| crate::builtins::common::map_control_flow_with_builtin(err, NAME))?;
        }
        let outside = !xq.is_finite()
            || !yq.is_finite()
            || xq < parsed.x_axis[0]
            || xq > parsed.x_axis[parsed.x_axis.len() - 1]
            || yq < parsed.y_axis[0]
            || yq > parsed.y_axis[parsed.y_axis.len() - 1];
        let sample = |spline: &TensorProductSpline2D, imaginary_lane: bool| {
            if !xq.is_finite() || !yq.is_finite() {
                f64::NAN
            } else if outside && !matches!(parsed.extrap, Extrapolation::Extrapolate) {
                if imaginary_lane && matches!(parsed.extrap, Extrapolation::Value(_)) {
                    0.0
                } else {
                    out_of_range_value(&parsed.extrap)
                }
            } else {
                spline.evaluate(&parsed.x_axis, &parsed.y_axis, xq, yq, |offset| {
                    parsed.z.at(offset, imaginary_lane)
                })
            }
        };
        let re = sample(&real, false);
        let im = imaginary.as_ref().map(|spline| sample(spline, true));
        output.set(index, re, im)?;
    }
    let result = output.finish(parsed)?;
    if let (Some(receipt), Some(runtime)) = (receipt, runtime) {
        receipt
            .checkpoint(runtime)
            .map_err(|error| super::super::pp::host_allocation_error(NAME, error))?;
    }
    Ok(result)
}

enum SplineOutput {
    Scalar(Option<f64>),
    Real(Vec<f64>),
    Complex(Vec<(f64, f64)>),
    ExactReal(ExactHostTensorAllocation),
    ExactComplex(ExactHostComplexTensorAllocation),
}

impl SplineOutput {
    fn new(
        parsed: &ParsedInterp2,
        receipt: Option<&HostAllocationReceipt>,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> crate::BuiltinResult<Self> {
        let count = parsed.xq.len();
        let dtype = parsed.z.dtype();
        if let (Some(receipt), Some(runtime)) = (receipt, runtime) {
            if parsed.z.is_complex() {
                let output = receipt
                    .with_output_allocation(runtime, |permit| {
                        ExactHostComplexTensorAllocation::new(dtype, &parsed.output_shape, permit)
                    })
                    .map_err(|error| super::super::pp::host_allocation_error(NAME, error))?;
                return Ok(Self::ExactComplex(output));
            }
            if count != 1 || dtype != NumericDType::F64 {
                let output = receipt
                    .with_output_allocation(runtime, |permit| {
                        ExactHostTensorAllocation::new(dtype, &parsed.output_shape, permit)
                    })
                    .map_err(|error| super::super::pp::host_allocation_error(NAME, error))?;
                return Ok(Self::ExactReal(output));
            }
            return Ok(Self::Scalar(None));
        }
        if parsed.z.is_complex() {
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| interp2_invalid_input("insufficient complex spline output memory"))?;
            Ok(Self::Complex(values))
        } else {
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| interp2_invalid_input("insufficient spline output memory"))?;
            Ok(Self::Real(values))
        }
    }

    fn set(&mut self, index: usize, real: f64, imaginary: Option<f64>) -> crate::BuiltinResult<()> {
        match self {
            Self::Scalar(slot) => *slot = Some(real),
            Self::Real(values) => values.push(real),
            Self::Complex(values) => values.push((real, imaginary.expect("complex spline lane"))),
            Self::ExactReal(values) => match values.as_f64_slice_mut() {
                Some(slice) => slice[index] = real,
                None => {
                    values.as_f32_slice_mut().expect("single spline output")[index] = real as f32
                }
            },
            Self::ExactComplex(values) => values
                .set(index, real, imaginary.expect("complex spline lane"))
                .map_err(|error| interp2_invalid_input(format!("complex output: {error}")))?,
        }
        Ok(())
    }

    fn finish(self, parsed: &ParsedInterp2) -> crate::BuiltinResult<Value> {
        let fail = |error: String| {
            interp2_error_with_message(format!("{NAME}: {error}"), &INTERP2_ERROR_INTERNAL)
        };
        match self {
            Self::Scalar(Some(value)) => Ok(Value::Num(value)),
            Self::Scalar(None) => Err(fail("scalar spline output was not filled".into())),
            Self::Real(values) if values.len() == 1 && parsed.z.dtype() == NumericDType::F64 => {
                Ok(Value::Num(values[0]))
            }
            Self::Real(values) => {
                Tensor::new_with_dtype(values, parsed.output_shape.clone(), parsed.z.dtype())
                    .map(Value::Tensor)
                    .map_err(fail)
            }
            Self::Complex(values) => ComplexTensor::from_f64_values_with_dtype(
                values,
                parsed.output_shape.clone(),
                parsed.z.dtype(),
            )
            .map(Value::ComplexTensor)
            .map_err(fail),
            Self::ExactReal(values) => values
                .into_tensor()
                .map(Value::Tensor)
                .map_err(|error| fail(error.to_string())),
            Self::ExactComplex(values) => values
                .into_tensor()
                .map(Value::ComplexTensor)
                .map_err(|error| fail(error.to_string())),
        }
    }
}

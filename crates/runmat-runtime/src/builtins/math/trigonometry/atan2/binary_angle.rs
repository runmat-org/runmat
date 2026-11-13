//! Borrowed numeric execution shared by radian and degree four-quadrant angles.

use runmat_value::{IntegerStorage, NumericDType, NumericStorage, Tensor};

use crate::builtins::common::broadcast::BroadcastPlan;

use super::{matlab_atan2_f32, matlab_atan2_f64};

#[derive(Clone, Copy)]
pub(in crate::builtins::math::trigonometry) enum Unit {
    Radians,
    Degrees,
}

pub(in crate::builtins::math::trigonometry) enum ComputeError {
    SizeMismatch(String),
    Internal(String),
}

enum BorrowedReal<'a> {
    Double(&'a [f64]),
    Single(&'a [f32]),
    Integer(&'a IntegerStorage),
}

impl<'a> BorrowedReal<'a> {
    fn new(tensor: &'a Tensor) -> Self {
        match tensor.numeric_dtype() {
            NumericDType::F64 => Self::Double(
                tensor
                    .as_f64_slice()
                    .expect("double tensor has native double storage"),
            ),
            NumericDType::F32 => Self::Single(
                tensor
                    .as_f32_slice()
                    .expect("single tensor has native single storage"),
            ),
            NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::I64
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32
            | NumericDType::U64 => Self::Integer(
                tensor
                    .integer_storage()
                    .expect("integer tensor has native integer storage"),
            ),
        }
    }

    fn f64_at(&self, index: usize) -> f64 {
        match self {
            Self::Double(values) => values[index],
            Self::Single(values) => f64::from(values[index]),
            Self::Integer(values) => values
                .value_at(index)
                .expect("broadcast index is within integer storage")
                .to_f64(),
        }
    }

    fn f32_at(&self, index: usize) -> f32 {
        let Self::Single(values) = self else {
            unreachable!("native-single evaluation requires single inputs")
        };
        values[index]
    }
}

impl Unit {
    pub(in crate::builtins::math::trigonometry) fn f64(self, y: f64, x: f64) -> f64 {
        let radians = matlab_atan2_f64(y, x);
        match self {
            Self::Radians => radians,
            Self::Degrees => radians * (180.0 / std::f64::consts::PI),
        }
    }

    pub(in crate::builtins::math::trigonometry) fn f32(self, y: f32, x: f32) -> f32 {
        let radians = matlab_atan2_f32(y, x);
        match self {
            Self::Radians => radians,
            Self::Degrees => radians * (180.0 / std::f32::consts::PI),
        }
    }
}

pub(in crate::builtins::math::trigonometry) fn compute(
    y: &Tensor,
    x: &Tensor,
    unit: Unit,
) -> Result<Tensor, ComputeError> {
    let plan = BroadcastPlan::new(y.shape(), x.shape()).map_err(ComputeError::SizeMismatch)?;
    let y_values = BorrowedReal::new(y);
    let x_values = BorrowedReal::new(x);
    let y_single = y.numeric_dtype() == NumericDType::F32;
    let x_single = x.numeric_dtype() == NumericDType::F32;
    let storage = if y_single || x_single {
        let mut output = vec![0.0_f32; plan.len()];
        for (index, yi, xi) in plan.iter() {
            output[index] = if y_single && x_single {
                unit.f32(y_values.f32_at(yi), x_values.f32_at(xi))
            } else {
                unit.f64(y_values.f64_at(yi), x_values.f64_at(xi)) as f32
            };
        }
        NumericStorage::F32(output)
    } else {
        let mut output = vec![0.0_f64; plan.len()];
        for (index, yi, xi) in plan.iter() {
            output[index] = unit.f64(y_values.f64_at(yi), x_values.f64_at(xi));
        }
        NumericStorage::F64(output)
    };
    Tensor::from_numeric_storage(storage, plan.output_shape().to_vec())
        .map_err(ComputeError::Internal)
}

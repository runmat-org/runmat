//! Host materialization at an admitted numerical boundary. Integer components
//! remain exact until a mathematical operation needs floating-point values.
use super::numerics::ScaledMagnitude;
use super::options::{invalid, ReductionKind};
use crate::builtins::common::tensor;
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_value::{NumericDType, NumericScalar, Tensor, Value};

#[derive(Clone, Copy)]
pub(super) enum Component {
    Floating(f64),
    Integer(i128),
}

impl Component {
    pub(super) fn value(self) -> f64 {
        match self {
            Self::Floating(value) => value,
            Self::Integer(value) => value as f64,
        }
    }

    fn normalized_difference(self, other: Self, scale: f64) -> f64 {
        match (self, other) {
            (Self::Integer(left), Self::Integer(right)) => (left - right) as f64 / scale,
            (Self::Integer(left), Self::Floating(right)) => {
                if let Some(right) = integral_i128(right) {
                    if let Some(difference) = left.checked_sub(right) {
                        return difference as f64 / scale;
                    }
                }
                self.value() / scale - other.value() / scale
            }
            (Self::Floating(left), Self::Integer(right)) => {
                if let Some(left) = integral_i128(left) {
                    if let Some(difference) = left.checked_sub(right) {
                        return difference as f64 / scale;
                    }
                }
                self.value() / scale - other.value() / scale
            }
            _ => self.value() / scale - other.value() / scale,
        }
    }
}

fn integral_i128(value: f64) -> Option<i128> {
    // The exclusive upper bound avoids Rust's saturating float-to-int cast at
    // 2^127, while the lower bound itself is exactly representable.
    (value.is_finite()
        && value.fract() == 0.0
        && value >= i128::MIN as f64
        && value < 2.0_f64.powi(127))
    .then_some(value as i128)
}

impl From<NumericScalar> for Component {
    fn from(value: NumericScalar) -> Self {
        match value {
            NumericScalar::F64(value) => Self::Floating(value),
            NumericScalar::F32(value) => Self::Floating(f64::from(value)),
            NumericScalar::I8(value) => Self::Integer(i128::from(value)),
            NumericScalar::I16(value) => Self::Integer(i128::from(value)),
            NumericScalar::I32(value) => Self::Integer(i128::from(value)),
            NumericScalar::I64(value) => Self::Integer(i128::from(value)),
            NumericScalar::U8(value) => Self::Integer(i128::from(value)),
            NumericScalar::U16(value) => Self::Integer(i128::from(value)),
            NumericScalar::U32(value) => Self::Integer(i128::from(value)),
            NumericScalar::U64(value) => Self::Integer(i128::from(value)),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Point {
    real: Component,
    imaginary: Component,
}

impl Point {
    fn real(value: Component) -> Self {
        Self {
            real: value,
            imaginary: Component::Integer(0),
        }
    }

    pub(super) fn scaled_magnitude(self) -> ScaledMagnitude {
        ScaledMagnitude::from_components(self.real.value(), self.imaginary.value())
    }

    pub(super) fn scaled_difference(self, other: Self) -> ScaledMagnitude {
        let components = [
            self.real.value(),
            self.imaginary.value(),
            other.real.value(),
            other.imaginary.value(),
        ];
        if components.iter().any(|component| !component.is_finite()) {
            return ScaledMagnitude::from_components(
                self.real.value() - other.real.value(),
                self.imaginary.value() - other.imaginary.value(),
            );
        }
        let scale = components
            .into_iter()
            .fold(0.0_f64, |largest, component| largest.max(component.abs()));
        if scale == 0.0 {
            return ScaledMagnitude::from_normalized(0.0, 0.0, 0.0);
        }
        ScaledMagnitude::from_normalized(
            scale,
            self.real.normalized_difference(other.real, scale),
            self.imaginary.normalized_difference(other.imaginary, scale),
        )
    }
}

pub(super) struct HostData {
    pub(super) points: Vec<Point>,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: NumericDType,
    pub(super) complex: bool,
}

impl HostData {
    pub(super) fn result_dtype(&self) -> NumericDType {
        if self.dtype == NumericDType::F32 {
            NumericDType::F32
        } else {
            NumericDType::F64
        }
    }

    pub(super) fn real_values(
        self,
        kind: ReductionKind,
    ) -> BuiltinResult<(Vec<f64>, Vec<usize>, NumericDType)> {
        if self.complex {
            return Err(invalid(kind, "input must be real"));
        }
        let dtype = self.result_dtype();
        let values = self
            .points
            .into_iter()
            .map(|point| point.real.value())
            .collect();
        Ok((values, self.shape, dtype))
    }

    pub(super) fn scaled_magnitudes(self) -> (Vec<ScaledMagnitude>, Vec<usize>, NumericDType) {
        let dtype = self.result_dtype();
        let values = self
            .points
            .into_iter()
            .map(Point::scaled_magnitude)
            .collect();
        (values, self.shape, dtype)
    }
}

pub(super) struct NonnegativeWeights {
    values: Vec<f64>,
    shape: Vec<usize>,
}

impl NonnegativeWeights {
    pub(super) fn from_host(data: HostData, kind: ReductionKind) -> BuiltinResult<Self> {
        let (values, shape, _) = data.real_values(kind)?;
        if values.iter().any(|value| *value < 0.0) {
            return Err(invalid(kind, "weights must be nonnegative"));
        }
        Ok(Self { values, shape })
    }

    pub(super) fn into_parts(self) -> (Vec<f64>, Vec<usize>) {
        (self.values, self.shape)
    }
}

pub(super) async fn materialize(kind: ReductionKind, value: Value) -> BuiltinResult<HostData> {
    let value = gather_if_needed_async(&value).await.map_err(|source| {
        let mut error = invalid(kind, "failed to materialize numeric input");
        error.source = Some(Box::new(source));
        error
    })?;
    match value {
        Value::Num(value) => Ok(HostData {
            points: vec![Point::real(Component::Floating(value))],
            shape: vec![1, 1],
            dtype: NumericDType::F64,
            complex: false,
        }),
        Value::Int(value) => Ok(HostData {
            points: vec![Point::real(Component::Integer(value.to_i128()))],
            shape: vec![1, 1],
            dtype: value.numeric_dtype(),
            complex: false,
        }),
        Value::Complex(real, imaginary) => Ok(HostData {
            points: vec![Point {
                real: Component::Floating(real),
                imaginary: Component::Floating(imaginary),
            }],
            shape: vec![1, 1],
            dtype: NumericDType::F64,
            complex: true,
        }),
        Value::ComplexTensor(tensor) => {
            let dtype = tensor.numeric_dtype();
            let shape = preserved_shape(tensor.shape(), tensor.len());
            let points = (0..tensor.len())
                .map(|index| {
                    let (real, imaginary) = tensor
                        .numeric_value_at(index)
                        .expect("complex storage index is in bounds");
                    Point {
                        real: real.into(),
                        imaginary: imaginary.into(),
                    }
                })
                .collect();
            Ok(HostData {
                points,
                shape,
                dtype,
                complex: true,
            })
        }
        Value::CharArray(chars) if kind == ReductionKind::RootMeanSquare => {
            let shape = preserved_shape(&chars.shape, chars.data().len());
            let points = chars
                .to_column_major()
                .into_iter()
                .map(|character| Point::real(Component::Floating(f64::from(character.raw()))))
                .collect();
            Ok(HostData {
                points,
                shape,
                dtype: NumericDType::F64,
                complex: false,
            })
        }
        other => {
            let input = tensor::value_into_tensor_for(kind.name(), other)
                .map_err(|source| invalid(kind, &source.to_string()))?;
            from_tensor(input)
        }
    }
}

fn from_tensor(input: Tensor) -> BuiltinResult<HostData> {
    let dtype = input.numeric_dtype();
    let shape = preserved_shape(input.shape(), input.len());
    let points = (0..input.len())
        .map(|index| {
            Point::real(Component::from(
                input
                    .numeric_value_at(index)
                    .expect("tensor storage index in bounds"),
            ))
        })
        .collect();
    Ok(HostData {
        points,
        shape,
        dtype,
        complex: false,
    })
}

fn preserved_shape(shape: &[usize], len: usize) -> Vec<usize> {
    if shape.is_empty() {
        tensor::default_shape_for(shape, len)
    } else {
        shape.to_vec()
    }
}

pub(super) fn integer_data(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.numeric_dtype().integer_class().is_some(),
        Value::ComplexTensor(tensor) => tensor.numeric_dtype().integer_class().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

pub(super) fn integer_control(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.numeric_dtype().integer_class().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

pub(super) fn exact_integer_weight(value: &Value) -> bool {
    integer_control(value)
}

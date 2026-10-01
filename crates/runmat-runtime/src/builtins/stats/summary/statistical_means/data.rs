//! Host materialization at an admitted numerical boundary. Integer components
//! remain exact until a mathematical operation needs floating-point values.
use super::options::{invalid, ReductionKind};
use crate::builtins::common::tensor;
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_value::{IntValue, NumericDType, NumericScalar, Tensor, Value};

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

    fn difference(self, other: Self) -> f64 {
        match (self, other) {
            (Self::Integer(left), Self::Integer(right)) => (left - right) as f64,
            _ => self.value() - other.value(),
        }
    }
}

impl From<NumericScalar> for Component {
    fn from(value: NumericScalar) -> Self {
        match value {
            NumericScalar::F64(value) => Self::Floating(value),
            NumericScalar::F32(value) => Self::Floating(f64::from(value)),
            integer => Self::Integer(
                integer
                    .into_int_value()
                    .expect("nonfloating numeric scalar is an integer")
                    .to_i128(),
            ),
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

    pub(super) fn magnitude(self) -> f64 {
        self.real.value().hypot(self.imaginary.value())
    }

    pub(super) fn difference_magnitude(self, other: Self) -> f64 {
        self.real
            .difference(other.real)
            .hypot(self.imaginary.difference(other.imaginary))
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

    pub(super) fn magnitudes(self) -> (Vec<f64>, Vec<usize>, NumericDType) {
        let dtype = self.result_dtype();
        let values = self.points.into_iter().map(Point::magnitude).collect();
        (values, self.shape, dtype)
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
            let shape = tensor::default_shape_for(&tensor.shape, tensor.len());
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
        other => {
            let input = tensor::value_into_tensor_for(kind.name(), other)
                .map_err(|source| invalid(kind, &source.to_string()))?;
            from_tensor(input)
        }
    }
}

fn from_tensor(input: Tensor) -> BuiltinResult<HostData> {
    let dtype = input.numeric_dtype();
    let shape = tensor::default_shape_for(&input.shape, input.len());
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

pub(super) fn integer_weight_nonnegative(value: &IntValue) -> bool {
    value.to_i128() >= 0
}

//! MATLAB single-precision signal policy: A, B, StartVector or Afun select
//! single if any is single. A scalar sigma changes domain, not precision.
use runmat_value::{NumericDType, Value};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Class {
    RealF32,
    RealF64,
    ComplexF32,
    ComplexF64,
}
impl Class {
    pub fn single(self) -> bool {
        matches!(self, Self::RealF32 | Self::ComplexF32)
    }
    pub fn complex(self) -> bool {
        matches!(self, Self::ComplexF32 | Self::ComplexF64)
    }
    pub fn of(value: &Value) -> Option<Self> {
        let (dtype, complex) = match value {
            Value::SparseTensor(s) => (
                s.numeric_dtype()?,
                s.as_complex_f32_slice().is_some() || s.as_complex_f64_slice().is_some(),
            ),
            Value::Tensor(t) => (t.numeric_dtype(), false),
            Value::ComplexTensor(t) => (t.numeric_dtype(), true),
            Value::Complex(..) => (NumericDType::F64, true),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => {
                (NumericDType::F64, false)
            }
            _ => return None,
        };
        Some(match (dtype, complex) {
            (NumericDType::F32, false) => Self::RealF32,
            (NumericDType::F32, true) => Self::ComplexF32,
            (_, false) => Self::RealF64,
            (_, true) => Self::ComplexF64,
        })
    }
    pub fn combine(self, other: Self) -> Self {
        match (
            self.single() || other.single(),
            self.complex() || other.complex(),
        ) {
            (true, false) => Self::RealF32,
            (false, false) => Self::RealF64,
            (true, true) => Self::ComplexF32,
            (false, true) => Self::ComplexF64,
        }
    }
    pub fn request(a: &Value, request: &super::request::Request<'_>) -> Self {
        let mut class = Self::of(a).unwrap_or(Self::RealF64);
        for value in [request.b, request.start].into_iter().flatten() {
            if let Some(other) = Self::of(value) {
                class = class.combine(other);
            }
        }
        if matches!(request.sigma,super::super::Sigma::Near(value) if value.im!=0.0) {
            class = class.combine(if class.single() {
                Self::ComplexF32
            } else {
                Self::ComplexF64
            });
        }
        class
    }
}

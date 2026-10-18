use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_value::{ComplexElement, ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::math::linalg::factor::schur::SchurForm;
use crate::context::RuntimeContext;
use crate::BuiltinResult;

use super::{checkpoint, invalid, numerical, unsupported_provider};

pub(super) enum VectorRoots<'a> {
    RealF32(&'a [f32]),
    RealF64(&'a [f64]),
    ComplexF32(&'a [ComplexElement<f32>]),
    ComplexF64(&'a [ComplexElement<f64>]),
    ComplexScalar(f64, f64),
}

impl VectorRoots<'_> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::RealF32(values) => values.len(),
            Self::RealF64(values) => values.len(),
            Self::ComplexF32(values) => values.len(),
            Self::ComplexF64(values) => values.len(),
            Self::ComplexScalar(..) => 1,
        }
    }

    pub(super) fn class(&self) -> (NumericDType, bool) {
        match self {
            Self::RealF32(_) => (NumericDType::F32, false),
            Self::RealF64(_) => (NumericDType::F64, false),
            Self::ComplexF32(_) => (NumericDType::F32, true),
            Self::ComplexF64(_) | Self::ComplexScalar(..) => (NumericDType::F64, true),
        }
    }
}

pub(super) enum MatrixInput<'a> {
    Real(&'a Tensor),
    Complex(&'a ComplexTensor),
}

pub(super) enum PolyInput<'a> {
    Vector(VectorRoots<'a>),
    Matrix(MatrixInput<'a>),
}

impl<'a> PolyInput<'a> {
    pub(super) fn classify(value: &'a Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(value) => Ok(Self::Vector(VectorRoots::RealF64(std::slice::from_ref(
                value,
            )))),
            Value::Complex(real, imag) => {
                Ok(Self::Vector(VectorRoots::ComplexScalar(*real, *imag)))
            }
            Value::Tensor(tensor) => {
                let shape = tensor.shape();
                validate_shape_and_class(shape, tensor.numeric_dtype())?;
                if is_vector(shape) {
                    let roots = match tensor.numeric_dtype() {
                        NumericDType::F32 => VectorRoots::RealF32(
                            tensor
                                .as_f32_slice()
                                .ok_or_else(|| invalid("invalid single storage"))?,
                        ),
                        NumericDType::F64 => VectorRoots::RealF64(
                            tensor
                                .as_f64_slice()
                                .ok_or_else(|| invalid("invalid double storage"))?,
                        ),
                        _ => unreachable!("class checked above"),
                    };
                    Ok(Self::Vector(roots))
                } else {
                    Ok(Self::Matrix(MatrixInput::Real(tensor)))
                }
            }
            Value::ComplexTensor(tensor) => {
                let shape = tensor.shape();
                validate_shape_and_class(shape, tensor.numeric_dtype())?;
                if is_vector(shape) {
                    let roots = match tensor.numeric_dtype() {
                        NumericDType::F32 => VectorRoots::ComplexF32(
                            tensor
                                .as_f32_slice()
                                .ok_or_else(|| invalid("invalid complex single storage"))?,
                        ),
                        NumericDType::F64 => VectorRoots::ComplexF64(
                            tensor
                                .as_f64_slice()
                                .ok_or_else(|| invalid("invalid complex double storage"))?,
                        ),
                        _ => unreachable!("class checked above"),
                    };
                    Ok(Self::Vector(roots))
                } else {
                    Ok(Self::Matrix(MatrixInput::Complex(tensor)))
                }
            }
            Value::GpuTensor(_) => Err(unsupported_provider(
                "poly has no admitted provider-output restoration for device input",
            )),
            _ => Err(invalid("poly requires single or double real/complex input")),
        }
    }
}

fn is_vector(shape: &[usize]) -> bool {
    shape.len() <= 2 && (shape.len() < 2 || shape[0] == 1 || shape[1] == 1 || shape == [0, 0])
}

fn validate_shape_and_class(shape: &[usize], dtype: NumericDType) -> BuiltinResult<()> {
    if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(invalid("poly requires single or double real/complex input"));
    }
    if !is_vector(shape) && (shape.len() != 2 || shape[0] != shape[1]) {
        return Err(invalid(
            "poly requires a vector or square two-dimensional matrix",
        ));
    }
    Ok(())
}

pub(super) fn real_schur(
    tensor: &Tensor,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<SchurForm> {
    let n = tensor.rows();
    let matrix = match tensor.numeric_dtype() {
        NumericDType::F32 => {
            let values = tensor
                .as_f32_slice()
                .ok_or_else(|| invalid("invalid single storage"))?;
            validate_finite(values, context, |value| value.is_finite())?;
            DMatrix::from_iterator(n, n, values.iter().copied().map(f64::from))
        }
        NumericDType::F64 => {
            let values = tensor
                .as_f64_slice()
                .ok_or_else(|| invalid("invalid double storage"))?;
            validate_finite(values, context, |value| value.is_finite())?;
            DMatrix::from_iterator(n, n, values.iter().copied())
        }
        _ => unreachable!("class checked before Schur"),
    };
    checkpoint(context)?;
    SchurForm::real(matrix).map_err(numerical)
}

pub(super) fn complex_schur(
    tensor: &ComplexTensor,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<SchurForm> {
    let n = tensor.rows();
    let matrix = match tensor.numeric_dtype() {
        NumericDType::F32 => {
            let values = tensor
                .as_f32_slice()
                .ok_or_else(|| invalid("invalid complex single storage"))?;
            validate_finite(values, context, |value| {
                value.0.is_finite() && value.1.is_finite()
            })?;
            DMatrix::from_iterator(
                n,
                n,
                values
                    .iter()
                    .map(|value| Complex64::new(f64::from(value.0), f64::from(value.1))),
            )
        }
        NumericDType::F64 => {
            let values = tensor
                .as_f64_slice()
                .ok_or_else(|| invalid("invalid complex double storage"))?;
            validate_finite(values, context, |value| {
                value.0.is_finite() && value.1.is_finite()
            })?;
            DMatrix::from_iterator(
                n,
                n,
                values.iter().map(|value| Complex64::new(value.0, value.1)),
            )
        }
        _ => unreachable!("class checked before Schur"),
    };
    checkpoint(context)?;
    SchurForm::complex(matrix).map_err(numerical)
}

fn validate_finite<T>(
    values: &[T],
    context: Option<&RuntimeContext>,
    finite: impl Fn(&T) -> bool,
) -> BuiltinResult<()> {
    for chunk in values.chunks(1024) {
        checkpoint(context)?;
        if !chunk.iter().all(|value| finite(value)) {
            return Err(invalid("poly input must be finite"));
        }
    }
    Ok(())
}

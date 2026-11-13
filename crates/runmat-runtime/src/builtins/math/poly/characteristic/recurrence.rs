//! In-place descending-coefficient updates on the sole final output owner.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_value::{ComplexElement, NumericDType, Value};

use crate::context::RuntimeContext;
use crate::BuiltinResult;

use super::input::VectorRoots;
use super::{checkpoint, invalid, output};

pub(super) fn vector(
    value: &mut Value,
    roots: VectorRoots<'_>,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    match roots {
        VectorRoots::RealF32(roots) => {
            let coefficients = output::real_f32(value)?;
            coefficients[0] = 1.0;
            for (degree, &root) in roots.iter().enumerate() {
                check_root(root.is_finite(), context)?;
                linear_f32(coefficients, degree, root, context)?;
            }
        }
        VectorRoots::RealF64(roots) => {
            let coefficients = output::real_f64(value)?;
            coefficients[0] = 1.0;
            for (degree, &root) in roots.iter().enumerate() {
                check_root(root.is_finite(), context)?;
                linear_f64(coefficients, degree, root, context)?;
            }
        }
        VectorRoots::ComplexF32(roots) => {
            let coefficients = output::complex_f32(value)?;
            coefficients[0] = ComplexElement(1.0, 0.0);
            for (degree, &ComplexElement(real, imag)) in roots.iter().enumerate() {
                check_root(real.is_finite() && imag.is_finite(), context)?;
                complex_native_f32(coefficients, degree, real, imag, context)?;
            }
        }
        VectorRoots::ComplexF64(roots) => {
            let coefficients = output::complex_f64(value)?;
            coefficients[0] = ComplexElement(1.0, 0.0);
            for (degree, &ComplexElement(real, imag)) in roots.iter().enumerate() {
                check_root(real.is_finite() && imag.is_finite(), context)?;
                complex_f64(coefficients, degree, real, imag, context)?;
            }
        }
        VectorRoots::ComplexScalar(real, imag) => {
            let coefficients = output::complex_f64(value)?;
            coefficients[0] = ComplexElement(1.0, 0.0);
            check_root(real.is_finite() && imag.is_finite(), context)?;
            complex_f64(coefficients, 0, real, imag, context)?;
        }
    }
    Ok(())
}

fn check_root(finite: bool, context: Option<&RuntimeContext>) -> BuiltinResult<()> {
    checkpoint(context)?;
    if finite {
        Ok(())
    } else {
        Err(invalid("poly input must be finite"))
    }
}

fn inner_checkpoint(index: usize, context: Option<&RuntimeContext>) -> BuiltinResult<()> {
    if index.is_multiple_of(1024) {
        checkpoint(context)?;
    }
    Ok(())
}

fn linear_f32(
    coefficients: &mut [f32],
    degree: usize,
    root: f32,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    for index in (1..=degree + 1).rev() {
        inner_checkpoint(index, context)?;
        coefficients[index] -= root * coefficients[index - 1];
    }
    Ok(())
}

// The canonical matrix Schur owner computes in f64 even for a single source.
// Delay narrowing its factor until the final coefficient update.
fn linear_schur_f32(
    coefficients: &mut [f32],
    degree: usize,
    root: f64,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    for index in (1..=degree + 1).rev() {
        inner_checkpoint(index, context)?;
        coefficients[index] =
            (f64::from(coefficients[index]) - root * f64::from(coefficients[index - 1])) as f32;
    }
    Ok(())
}

fn linear_f64(
    coefficients: &mut [f64],
    degree: usize,
    root: f64,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    for index in (1..=degree + 1).rev() {
        inner_checkpoint(index, context)?;
        coefficients[index] -= root * coefficients[index - 1];
    }
    Ok(())
}

fn complex_f32(
    coefficients: &mut [ComplexElement<f32>],
    degree: usize,
    real: f64,
    imag: f64,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    for index in (1..=degree + 1).rev() {
        inner_checkpoint(index, context)?;
        let ComplexElement(previous_real, previous_imag) = coefficients[index - 1];
        let ComplexElement(current_real, current_imag) = coefficients[index];
        coefficients[index] = ComplexElement(
            (f64::from(current_real) - real * f64::from(previous_real)
                + imag * f64::from(previous_imag)) as f32,
            (f64::from(current_imag)
                - real * f64::from(previous_imag)
                - imag * f64::from(previous_real)) as f32,
        );
    }
    Ok(())
}

fn complex_native_f32(
    coefficients: &mut [ComplexElement<f32>],
    degree: usize,
    real: f32,
    imag: f32,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    for index in (1..=degree + 1).rev() {
        inner_checkpoint(index, context)?;
        let ComplexElement(previous_real, previous_imag) = coefficients[index - 1];
        let ComplexElement(current_real, current_imag) = coefficients[index];
        coefficients[index] = ComplexElement(
            current_real - real * previous_real + imag * previous_imag,
            current_imag - real * previous_imag - imag * previous_real,
        );
    }
    Ok(())
}

fn complex_f64(
    coefficients: &mut [ComplexElement<f64>],
    degree: usize,
    real: f64,
    imag: f64,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    for index in (1..=degree + 1).rev() {
        inner_checkpoint(index, context)?;
        let ComplexElement(previous_real, previous_imag) = coefficients[index - 1];
        let ComplexElement(current_real, current_imag) = coefficients[index];
        coefficients[index] = ComplexElement(
            current_real - real * previous_real + imag * previous_imag,
            current_imag - real * previous_imag - imag * previous_real,
        );
    }
    Ok(())
}

pub(super) fn real_schur(
    value: &mut Value,
    triangular: &DMatrix<f64>,
    dtype: NumericDType,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    match dtype {
        NumericDType::F32 => {
            let coefficients = output::real_f32(value)?;
            coefficients[0] = 1.0;
            let mut degree = 0;
            while degree < triangular.nrows() {
                checkpoint(context)?;
                if degree + 1 < triangular.nrows() && triangular[(degree + 1, degree)] != 0.0 {
                    let a = triangular[(degree, degree)];
                    let b = triangular[(degree, degree + 1)];
                    let c = triangular[(degree + 1, degree)];
                    let d = triangular[(degree + 1, degree + 1)];
                    quadratic_f32(coefficients, degree, a + d, a * d - b * c, context)?;
                    degree += 2;
                } else {
                    linear_schur_f32(coefficients, degree, triangular[(degree, degree)], context)?;
                    degree += 1;
                }
            }
        }
        NumericDType::F64 => {
            let coefficients = output::real_f64(value)?;
            coefficients[0] = 1.0;
            let mut degree = 0;
            while degree < triangular.nrows() {
                checkpoint(context)?;
                if degree + 1 < triangular.nrows() && triangular[(degree + 1, degree)] != 0.0 {
                    let a = triangular[(degree, degree)];
                    let b = triangular[(degree, degree + 1)];
                    let c = triangular[(degree + 1, degree)];
                    let d = triangular[(degree + 1, degree + 1)];
                    quadratic_f64(coefficients, degree, a + d, a * d - b * c, context)?;
                    degree += 2;
                } else {
                    linear_f64(coefficients, degree, triangular[(degree, degree)], context)?;
                    degree += 1;
                }
            }
        }
        _ => return Err(invalid("poly real output storage is invalid")),
    }
    Ok(())
}

fn quadratic_f32(
    coefficients: &mut [f32],
    degree: usize,
    trace: f64,
    determinant: f64,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    for index in (1..=degree + 2).rev() {
        inner_checkpoint(index, context)?;
        let old = if index <= degree {
            f64::from(coefficients[index])
        } else {
            0.0
        };
        let previous = if index - 1 <= degree {
            f64::from(coefficients[index - 1])
        } else {
            0.0
        };
        let before_previous = if index >= 2 {
            f64::from(coefficients[index - 2])
        } else {
            0.0
        };
        coefficients[index] = (old - trace * previous + determinant * before_previous) as f32;
    }
    Ok(())
}

fn quadratic_f64(
    coefficients: &mut [f64],
    degree: usize,
    trace: f64,
    determinant: f64,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    for index in (1..=degree + 2).rev() {
        inner_checkpoint(index, context)?;
        let old = if index <= degree {
            coefficients[index]
        } else {
            0.0
        };
        let previous = if index - 1 <= degree {
            coefficients[index - 1]
        } else {
            0.0
        };
        let before_previous = if index >= 2 {
            coefficients[index - 2]
        } else {
            0.0
        };
        coefficients[index] = old - trace * previous + determinant * before_previous;
    }
    Ok(())
}

pub(super) fn complex_schur(
    value: &mut Value,
    triangular: &DMatrix<Complex64>,
    dtype: NumericDType,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    match dtype {
        NumericDType::F32 => {
            let coefficients = output::complex_f32(value)?;
            coefficients[0] = ComplexElement(1.0, 0.0);
            for degree in 0..triangular.nrows() {
                checkpoint(context)?;
                let root = triangular[(degree, degree)];
                complex_f32(coefficients, degree, root.re, root.im, context)?;
            }
        }
        NumericDType::F64 => {
            let coefficients = output::complex_f64(value)?;
            coefficients[0] = ComplexElement(1.0, 0.0);
            for degree in 0..triangular.nrows() {
                checkpoint(context)?;
                let root = triangular[(degree, degree)];
                complex_f64(coefficients, degree, root.re, root.im, context)?;
            }
        }
        _ => return Err(invalid("poly complex output storage is invalid")),
    }
    Ok(())
}

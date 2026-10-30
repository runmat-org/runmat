//! Ordinary Schur factorization and transformations of retained Schur forms.

mod form;
mod ordschur;
#[cfg(target_arch = "wasm32")]
pub(crate) use ordschur::__runmat_wasm_register_builtin_ordschur_builtin;
mod reorder;
#[cfg(test)]
mod tests;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

pub(crate) use form::SchurForm;

#[derive(Clone, Copy)]
enum SchurMode {
    Real,
    Complex,
}

#[derive(Clone, Copy)]
struct SchurContext {
    name: &'static str,
    invalid_identifier: &'static str,
    argument_identifier: &'static str,
    numerical_identifier: &'static str,
}

const SCHUR: SchurContext = SchurContext {
    name: "schur",
    invalid_identifier: "RunMat:schur:InvalidInput",
    argument_identifier: "RunMat:schur:InvalidArgument",
    numerical_identifier: "RunMat:schur:Numerical",
};
const RSF2CSF: SchurContext = SchurContext {
    name: "rsf2csf",
    invalid_identifier: "RunMat:rsf2csf:InvalidInput",
    argument_identifier: "RunMat:rsf2csf:InvalidInput",
    numerical_identifier: "RunMat:rsf2csf:Numerical",
};
const ORDEIG: SchurContext = SchurContext {
    name: "ordeig",
    invalid_identifier: "RunMat:ordeig:InvalidInput",
    argument_identifier: "RunMat:ordeig:InvalidInput",
    numerical_identifier: "RunMat:ordeig:InvalidInput",
};
const ORDSCHUR: SchurContext = SchurContext {
    name: "ordschur",
    invalid_identifier: "RunMat:ordschur:InvalidInput",
    argument_identifier: "RunMat:ordschur:InvalidOutputs",
    numerical_identifier: "RunMat:ordschur:IllConditioned",
};

fn failure(context: SchurContext, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(context.name)
        .with_identifier(context.numerical_identifier)
        .build()
}

fn invalid(context: SchurContext, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(context.name)
        .with_identifier(context.invalid_identifier)
        .build()
}

fn invalid_argument(context: SchurContext, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(context.name)
        .with_identifier(context.argument_identifier)
        .build()
}

fn precision(value: &Value, name: SchurContext) -> BuiltinResult<NumericDType> {
    let dtype = match value {
        Value::Tensor(value) => value.numeric_dtype(),
        Value::ComplexTensor(value) => value.numeric_dtype(),
        Value::Num(_) | Value::Complex(_, _) => NumericDType::F64,
        _ => return Err(invalid(name, "expected a dense single or double matrix")),
    };
    if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(invalid(name, "expected a dense single or double matrix"));
    }
    Ok(dtype)
}

fn real_matrix(value: Value, name: SchurContext) -> BuiltinResult<DMatrix<f64>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.shape().len() > 2 || tensor.rows() != tensor.cols() {
                return Err(invalid(name, "expected a square two-dimensional matrix"));
            }
            let values = tensor.materialize_f64();
            if !values.iter().all(|v| v.is_finite()) {
                return Err(invalid(name, "matrix coefficients must be finite"));
            }
            Ok(DMatrix::from_column_slice(
                tensor.rows(),
                tensor.cols(),
                &values,
            ))
        }
        Value::Num(number) if number.is_finite() => Ok(DMatrix::from_element(1, 1, number)),
        _ => Err(invalid(
            name,
            "expected a real square single or double matrix",
        )),
    }
}

fn complex_matrix(value: Value, name: SchurContext) -> BuiltinResult<DMatrix<Complex64>> {
    match value {
        Value::ComplexTensor(tensor) => {
            if tensor.shape().len() > 2 || tensor.rows() != tensor.cols() {
                return Err(invalid(name, "expected a square two-dimensional matrix"));
            }
            let values = tensor.materialize_f64();
            if !values
                .iter()
                .all(|&(re, im)| re.is_finite() && im.is_finite())
            {
                return Err(invalid(name, "matrix coefficients must be finite"));
            }
            let values = values
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect::<Vec<_>>();
            Ok(DMatrix::from_column_slice(
                tensor.rows(),
                tensor.cols(),
                &values,
            ))
        }
        Value::Complex(re, im) if re.is_finite() && im.is_finite() => {
            Ok(DMatrix::from_element(1, 1, Complex64::new(re, im)))
        }
        real => Ok(real_matrix(real, name)?.map(|x| Complex64::new(x, 0.0))),
    }
}

fn real_value(
    matrix: &DMatrix<f64>,
    dtype: NumericDType,
    name: SchurContext,
) -> BuiltinResult<Value> {
    let shape = vec![matrix.nrows(), matrix.ncols()];
    if dtype == NumericDType::F32 {
        Tensor::from_f32(matrix.iter().map(|&value| value as f32).collect(), shape)
            .map(Value::Tensor)
            .map_err(|error| failure(name, error.to_string()))
    } else {
        Tensor::new(matrix.iter().copied().collect(), shape)
            .map(Value::Tensor)
            .map_err(|error| failure(name, error.to_string()))
    }
}

fn complex_value(
    matrix: &DMatrix<Complex64>,
    dtype: NumericDType,
    name: SchurContext,
) -> BuiltinResult<Value> {
    let shape = vec![matrix.nrows(), matrix.ncols()];
    if dtype == NumericDType::F32 {
        ComplexTensor::from_f32(
            matrix
                .iter()
                .map(|value| (value.re as f32, value.im as f32))
                .collect(),
            shape,
        )
        .map(Value::ComplexTensor)
        .map_err(|error| failure(name, error.to_string()))
    } else {
        ComplexTensor::new(
            matrix.iter().map(|value| (value.re, value.im)).collect(),
            shape,
        )
        .map(Value::ComplexTensor)
        .map_err(|error| failure(name, error.to_string()))
    }
}

fn output_pair(
    form: SchurForm,
    dtype: NumericDType,
    name: SchurContext,
    default_first: bool,
) -> BuiltinResult<Value> {
    let one = |first| match &form {
        SchurForm::Real {
            vectors,
            triangular,
        } => real_value(if first { vectors } else { triangular }, dtype, name),
        SchurForm::Complex {
            vectors,
            triangular,
        } => complex_value(if first { vectors } else { triangular }, dtype, name),
    };
    match crate::output_count::current_output_count() {
        None => one(default_first),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![one(default_first)?])),
        Some(2) => Ok(Value::output_list(vec![one(true)?, one(false)?])),
        Some(_) => Err(invalid(name, "at most two outputs are supported")),
    }
}

fn option_text(value: &Value, context: SchurContext) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => Some(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            )
        }
        Value::CharArray(chars) if chars.rows <= 1 => Some(
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = invalid_argument(context, source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}

fn validate_pair_outputs(context: SchurContext) -> BuiltinResult<()> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        Err(invalid_argument(
            context,
            "at most two outputs are supported",
        ))
    } else {
        Ok(())
    }
}

#[runtime_builtin(
    name = "schur",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::schur"
)]
async fn schur_builtin(matrix: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    const NAME: SchurContext = SCHUR;
    validate_pair_outputs(NAME)?;
    if rest.len() > 1 {
        return Err(invalid_argument(NAME, "schur accepts one optional mode"));
    }
    let dtype = precision(&matrix, NAME)?;
    let mode = if let Some(option) = rest.first() {
        let text = option_text(option, NAME)?
            .ok_or_else(|| invalid_argument(NAME, "mode must be 'real' or 'complex'"))?;
        if text.trim().eq_ignore_ascii_case("real") {
            SchurMode::Real
        } else if text.trim().eq_ignore_ascii_case("complex") {
            SchurMode::Complex
        } else {
            return Err(invalid_argument(NAME, "mode must be 'real' or 'complex'"));
        }
    } else {
        SchurMode::Real
    };
    let complex = matches!(&matrix, Value::ComplexTensor(_) | Value::Complex(_, _))
        || matches!(mode, SchurMode::Complex);
    let form = if complex {
        SchurForm::complex(complex_matrix(matrix, NAME)?)
    } else {
        SchurForm::real(real_matrix(matrix, NAME)?)
    }
    .map_err(|message| failure(NAME, message))?;
    output_pair(form, dtype, NAME, false)
}

#[runtime_builtin(
    name = "rsf2csf",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::schur"
)]
async fn rsf2csf_builtin(
    vectors: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    const NAME: SchurContext = RSF2CSF;
    validate_pair_outputs(NAME)?;
    let (1, Some(triangular)) = (rest.len(), rest.first()) else {
        return Err(invalid(NAME, "rsf2csf expects real Schur factors U and T"));
    };
    let dtype = precision(&vectors, NAME)?;
    if dtype != precision(triangular, NAME)? {
        return Err(invalid(
            NAME,
            "Schur factors must have the same floating class",
        ));
    }
    let form = SchurForm::from_real_parts(
        real_matrix(vectors, NAME)?,
        real_matrix(triangular.clone(), NAME)?,
    )
    .map_err(|message| invalid(NAME, message))?
    .into_complex()
    .map_err(|message| failure(NAME, message))?;
    output_pair(form, dtype, NAME, true)
}

#[runtime_builtin(
    name = "ordeig",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::schur",
    context = true
)]
async fn ordeig_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    triangular: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    const NAME: SchurContext = ORDEIG;
    if let (1, Some(second)) = (rest.len(), rest.first()) {
        return super::qz::generalized_ordeig(runtime.as_ref(), &triangular, second).await;
    }
    if !rest.is_empty() {
        return Err(invalid(
            NAME,
            "ordeig expects one ordinary or two generalized Schur factors",
        ));
    }
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(invalid(NAME, "ordeig returns one output"));
    }
    let dtype = precision(&triangular, NAME)?;
    let values = match triangular {
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            let matrix = complex_matrix(triangular, NAME)?;
            SchurForm::from_complex_parts(DMatrix::identity(matrix.nrows(), matrix.nrows()), matrix)
                .map_err(|message| invalid(NAME, message))?
                .eigenvalues()
        }
        real => {
            let matrix = real_matrix(real, NAME)?;
            SchurForm::from_real_parts(DMatrix::identity(matrix.nrows(), matrix.nrows()), matrix)
                .map_err(|message| invalid(NAME, message))?
                .eigenvalues()
        }
    };
    let n = values.len();
    let output = if values.iter().any(|value| value.im != 0.0) {
        complex_value(&DMatrix::from_column_slice(n, 1, &values), dtype, NAME)?
    } else {
        real_value(
            &DMatrix::from_column_slice(n, 1, &values.iter().map(|v| v.re).collect::<Vec<_>>()),
            dtype,
            NAME,
        )?
    };
    if crate::output_count::current_output_count() == Some(0) {
        Ok(Value::output_list(Vec::new()))
    } else if crate::output_count::current_output_count() == Some(1) {
        Ok(Value::output_list(vec![output]))
    } else {
        Ok(output)
    }
}

fn factor_from_values(
    vectors: Value,
    triangular: Value,
    name: SchurContext,
) -> BuiltinResult<(SchurForm, NumericDType)> {
    let dtype = precision(&vectors, name)?;
    if dtype != precision(&triangular, name)? {
        return Err(invalid(
            name,
            "Schur factors must have the same floating class",
        ));
    }
    let complex = matches!(&vectors, Value::ComplexTensor(_) | Value::Complex(_, _))
        || matches!(&triangular, Value::ComplexTensor(_) | Value::Complex(_, _));
    let form = if complex {
        SchurForm::from_complex_parts(
            complex_matrix(vectors, name)?,
            complex_matrix(triangular, name)?,
        )
    } else {
        SchurForm::from_real_parts(real_matrix(vectors, name)?, real_matrix(triangular, name)?)
    }
    .map_err(|message| invalid(name, message))?;
    Ok((form, dtype))
}

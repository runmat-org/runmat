//! State-space to SISO transfer-function conversion for control analysis.
use super::coefficients::clean_coefficients;
use crate::builtins::common::tensor;
use crate::builtins::control::tf_model::{
    control_error, gather_ss_properties, internal_identifier, invalid_model_identifier, poly_eval,
    scalar_f64, unsupported_model_identifier, TfModel, EPS,
};
use crate::{BuiltinResult, RuntimeError};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use runmat_value::{ObjectInstance, Value};

#[derive(Debug, Clone, Copy)]
enum ConversionErrorKind {
    InvalidModel,
    UnsupportedModel,
    Internal,
}
#[derive(Debug)]
pub(crate) struct ConversionError {
    kind: ConversionErrorKind,
    message: String,
}
type ConversionResult<T> = Result<T, ConversionError>;

fn conversion_error(message: impl Into<String>, kind: ConversionErrorKind) -> ConversionError {
    ConversionError {
        kind,
        message: message.into(),
    }
}
impl ConversionError {
    fn into_runtime(self, builtin: &'static str) -> RuntimeError {
        let identifier = match self.kind {
            ConversionErrorKind::InvalidModel => invalid_model_identifier(builtin),
            ConversionErrorKind::UnsupportedModel => unsupported_model_identifier(builtin),
            ConversionErrorKind::Internal => internal_identifier(builtin),
        };
        control_error(builtin, identifier, format!("{builtin}: {}", self.message))
    }
}

pub(crate) async fn ss_object_to_tf_async(
    object: ObjectInstance,
    builtin: &'static str,
) -> BuiltinResult<TfModel> {
    let gathered = gather_ss_properties(
        object,
        &["A", "B", "C", "D", "Ts", "InputDelay", "OutputDelay"],
        builtin,
    )
    .await?;
    ss_object_to_tf(&gathered, builtin).map_err(|error| error.into_runtime(builtin))
}

fn ss_object_to_tf(object: &ObjectInstance, builtin: &'static str) -> ConversionResult<TfModel> {
    let a = matrix_property(object, "A")?;
    let b = matrix_property(object, "B")?;
    let c = matrix_property(object, "C")?;
    let d = matrix_property(object, "D")?;
    let sample_time = scalar_property(object, "Ts", builtin)?;
    if !sample_time.is_finite() || sample_time < 0.0 {
        return Err(conversion_error(
            "ss sample time must be a finite nonnegative scalar",
            ConversionErrorKind::InvalidModel,
        ));
    }
    ensure_zero_delay_property(object, "InputDelay")?;
    ensure_zero_delay_property(object, "OutputDelay")?;
    validate_ss_dimensions(&a, &b, &c, &d)?;

    let denominator = characteristic_polynomial_from_matrix(&a)?;
    let numerator = state_space_numerator(&a, &b, &c, d[(0, 0)], &denominator)?;
    let numerator = trim_leading_complex_zeros(clean_coefficients(numerator));
    let denominator = trim_leading_complex_zeros(clean_coefficients(denominator));
    if denominator.is_empty() || denominator[0].norm() <= EPS {
        return Err(conversion_error(
            "ss model produced an invalid denominator polynomial",
            ConversionErrorKind::Internal,
        ));
    }
    ensure_finite_coefficients("Numerator", &numerator)?;
    ensure_finite_coefficients("Denominator", &denominator)?;

    Ok(TfModel {
        numerator,
        denominator,
        variable: if sample_time > 0.0 { "z" } else { "s" }.to_string(),
        sample_time,
        input_delay: 0.0,
        output_delay: 0.0,
    })
}

fn property<'a>(object: &'a ObjectInstance, name: &str) -> ConversionResult<&'a Value> {
    object.properties.get(name).ok_or_else(|| {
        conversion_error(
            format!("model object is missing {name} property"),
            ConversionErrorKind::InvalidModel,
        )
    })
}

fn matrix_property(object: &ObjectInstance, name: &str) -> ConversionResult<DMatrix<Complex64>> {
    let value = property(object, name)?;
    match value {
        Value::Tensor(tensor) => {
            ensure_matrix_shape(name, &tensor.shape)?;
            let data = tensor::tensor_values_f64(tensor)
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect::<Vec<_>>();
            ensure_finite_coefficients(name, &data)?;
            Ok(DMatrix::from_column_slice(tensor.rows, tensor.cols, &data))
        }
        Value::ComplexTensor(tensor) => {
            ensure_matrix_shape(name, &tensor.shape)?;
            let data = tensor::complex_tensor_values_complex64(tensor);
            ensure_finite_coefficients(name, &data)?;
            Ok(DMatrix::from_column_slice(tensor.rows, tensor.cols, &data))
        }
        Value::LogicalArray(logical) => {
            ensure_matrix_shape(name, &logical.shape)?;
            let (rows, cols) = rows_cols(&logical.shape);
            let data = logical
                .data
                .iter()
                .map(|&value| Complex64::new(if value == 0 { 0.0 } else { 1.0 }, 0.0))
                .collect::<Vec<_>>();
            Ok(DMatrix::from_column_slice(rows, cols, &data))
        }
        Value::Num(n) => scalar_matrix(*n, 0.0, name),
        Value::Int(i) => scalar_matrix(i.to_f64(), 0.0, name),
        Value::Bool(b) => scalar_matrix(if *b { 1.0 } else { 0.0 }, 0.0, name),
        Value::Complex(re, im) => scalar_matrix(*re, *im, name),
        other => Err(conversion_error(
            format!("ss {name} must be a finite numeric matrix, got {other:?}"),
            ConversionErrorKind::InvalidModel,
        )),
    }
}

fn scalar_matrix(re: f64, im: f64, name: &str) -> ConversionResult<DMatrix<Complex64>> {
    let value = Complex64::new(re, im);
    ensure_finite_coefficients(name, &[value])?;
    Ok(DMatrix::from_element(1, 1, value))
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> ConversionResult<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(conversion_error(
            format!("ss {name} must be a 2-D matrix, got shape {shape:?}"),
            ConversionErrorKind::InvalidModel,
        ))
    }
}

fn rows_cols(shape: &[usize]) -> (usize, usize) {
    if shape.len() >= 2 {
        (shape[0], shape[1])
    } else if shape.len() == 1 {
        (1, shape[0])
    } else {
        (0, 0)
    }
}

fn scalar_property(
    object: &ObjectInstance,
    name: &str,
    builtin: &'static str,
) -> ConversionResult<f64> {
    let value = scalar_f64(property(object, name)?, name, builtin)
        .map_err(|error| conversion_error(error.message(), ConversionErrorKind::InvalidModel))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(conversion_error(
            format!("ss {name} must be finite"),
            ConversionErrorKind::InvalidModel,
        ))
    }
}

fn ensure_zero_delay_property(object: &ObjectInstance, name: &str) -> ConversionResult<()> {
    let values = numeric_values(property(object, name)?, name)?;
    if values.iter().any(|value| value.norm() > EPS) {
        return Err(conversion_error(
            "ss models with input or output delays are not supported",
            ConversionErrorKind::UnsupportedModel,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage, Tensor};

    #[test]
    fn ss_conversion_preflights_then_selectively_downloads_nested_properties() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let state = Tensor::new(vec![-2.0], vec![1, 1]).unwrap();
            let unrelated = Tensor::new(vec![7.0, 8.0], vec![1, 2]).unwrap();
            let resident_state =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &state).unwrap();
            let resident_unrelated =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &unrelated).unwrap();

            for builtin in ["pzmap", "rlocus"] {
                let mut incomplete = ObjectInstance::new("ss".to_string());
                incomplete
                    .properties
                    .insert("A".into(), Value::GpuTensor(resident_state.clone()));
                provider.reset_telemetry();
                let error = block_on(ss_object_to_tf_async(incomplete, builtin))
                    .expect_err("missing properties must fail before download");
                assert_eq!(
                    error.identifier(),
                    Some(if builtin == "pzmap" {
                        "RunMat:pzmap:InvalidModel"
                    } else {
                        "RunMat:rlocus:InvalidModel"
                    })
                );
                assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

                let mut complete = ObjectInstance::new("ss".to_string());
                complete
                    .properties
                    .insert("A".into(), Value::GpuTensor(resident_state.clone()));
                for (name, value) in [
                    ("B", 1.0),
                    ("C", 1.0),
                    ("D", 0.0),
                    ("Ts", 0.0),
                    ("InputDelay", 0.0),
                    ("OutputDelay", 0.0),
                ] {
                    complete.properties.insert(name.into(), Value::Num(value));
                }
                complete.properties.insert(
                    "Unrelated".into(),
                    Value::GpuTensor(resident_unrelated.clone()),
                );
                provider.reset_telemetry();
                let model = block_on(ss_object_to_tf_async(complete, builtin))
                    .expect("resident state-space conversion");
                assert_eq!(
                    model.denominator,
                    vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
                );
                assert_eq!(provider.telemetry_snapshot().download_bytes, 8);
            }
        });
    }

    #[test]
    fn numeric_values_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-1, 0, 2]), vec![1, 3])
            .expect("typed integer tensor");

        assert_eq!(
            numeric_values(&Value::Tensor(tensor), "A").expect("numeric values"),
            vec![
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
            ]
        );
    }

    #[test]
    fn numeric_values_reads_complex_integer_storage_exactly() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, -2]),
            IntegerStorage::I16(vec![3, -4]),
        )
        .expect("complex integer storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();

        assert_eq!(
            numeric_values(&Value::ComplexTensor(tensor), "A").expect("numeric values"),
            vec![Complex64::new(1.0, 3.0), Complex64::new(-2.0, -4.0)]
        );
    }

    #[test]
    fn matrix_property_reads_typed_integer_storage_exactly() {
        let matrix = Tensor::new_integer(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2])
            .expect("typed integer matrix");
        let mut object = ObjectInstance::new("ss".to_string());
        object
            .properties
            .insert("A".to_string(), Value::Tensor(matrix));

        let parsed = matrix_property(&object, "A").expect("matrix property");

        assert_eq!(parsed[(0, 0)], Complex64::new(1.0, 0.0));
        assert_eq!(parsed[(1, 0)], Complex64::new(3.0, 0.0));
        assert_eq!(parsed[(0, 1)], Complex64::new(2.0, 0.0));
        assert_eq!(parsed[(1, 1)], Complex64::new(4.0, 0.0));
    }

    #[test]
    fn matrix_property_reads_complex_integer_storage_exactly() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 3, 2, 4]),
            IntegerStorage::I16(vec![-1, -3, -2, -4]),
        )
        .expect("complex integer storage");
        let matrix = ComplexTensor::new_integer(storage, vec![2, 2]).unwrap();
        let mut object = ObjectInstance::new("ss".to_string());
        object
            .properties
            .insert("A".to_string(), Value::ComplexTensor(matrix));

        let parsed = matrix_property(&object, "A").expect("matrix property");

        assert_eq!(parsed[(0, 0)], Complex64::new(1.0, -1.0));
        assert_eq!(parsed[(1, 0)], Complex64::new(3.0, -3.0));
        assert_eq!(parsed[(0, 1)], Complex64::new(2.0, -2.0));
        assert_eq!(parsed[(1, 1)], Complex64::new(4.0, -4.0));
    }
}

fn numeric_values(value: &Value, name: &str) -> ConversionResult<Vec<Complex64>> {
    let values = match value {
        Value::Num(n) => vec![Complex64::new(*n, 0.0)],
        Value::Int(i) => vec![Complex64::new(i.to_f64(), 0.0)],
        Value::Bool(b) => vec![Complex64::new(if *b { 1.0 } else { 0.0 }, 0.0)],
        Value::Complex(re, im) => vec![Complex64::new(*re, *im)],
        Value::Tensor(tensor) => tensor::tensor_values_f64(tensor)
            .into_iter()
            .map(|re| Complex64::new(re, 0.0))
            .collect(),
        Value::ComplexTensor(tensor) => tensor::complex_tensor_values_complex64(tensor),
        Value::LogicalArray(logical) => logical
            .data
            .iter()
            .map(|&value| Complex64::new(if value == 0 { 0.0 } else { 1.0 }, 0.0))
            .collect(),
        other => {
            return Err(conversion_error(
                format!("ss {name} must be numeric, got {other:?}"),
                ConversionErrorKind::InvalidModel,
            ));
        }
    };
    ensure_finite_coefficients(name, &values)?;
    Ok(values)
}

fn validate_ss_dimensions(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    c: &DMatrix<Complex64>,
    d: &DMatrix<Complex64>,
) -> ConversionResult<()> {
    if a.nrows() != a.ncols() {
        return Err(conversion_error(
            format!("ss A must be square, got {}x{}", a.nrows(), a.ncols()),
            ConversionErrorKind::InvalidModel,
        ));
    }
    let states = a.nrows();
    if b.nrows() != states {
        return Err(conversion_error(
            format!(
                "ss B must have {states} rows to match A, got {}x{}",
                b.nrows(),
                b.ncols()
            ),
            ConversionErrorKind::InvalidModel,
        ));
    }
    if c.ncols() != states {
        return Err(conversion_error(
            format!(
                "ss C must have {states} columns to match A, got {}x{}",
                c.nrows(),
                c.ncols()
            ),
            ConversionErrorKind::InvalidModel,
        ));
    }
    if d.nrows() != c.nrows() || d.ncols() != b.ncols() {
        return Err(conversion_error(
            format!(
                "ss D must have shape {}x{} to match C outputs and B inputs, got {}x{}",
                c.nrows(),
                b.ncols(),
                d.nrows(),
                d.ncols()
            ),
            ConversionErrorKind::InvalidModel,
        ));
    }
    if b.ncols() != 1 || c.nrows() != 1 || d.nrows() != 1 || d.ncols() != 1 {
        return Err(conversion_error(
            "only SISO ss models are supported",
            ConversionErrorKind::UnsupportedModel,
        ));
    }
    Ok(())
}

fn characteristic_polynomial_from_matrix(
    a: &DMatrix<Complex64>,
) -> ConversionResult<Vec<Complex64>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(vec![Complex64::new(1.0, 0.0)]);
    }
    let mut coeffs = Vec::with_capacity(n + 1);
    coeffs.push(Complex64::new(1.0, 0.0));
    let mut b = DMatrix::<Complex64>::identity(n, n);
    for k in 1..=n {
        let ab = a * &b;
        let trace = (0..n).fold(Complex64::new(0.0, 0.0), |acc, idx| acc + ab[(idx, idx)]);
        let coeff = -trace / Complex64::new(k as f64, 0.0);
        coeffs.push(coeff);
        let mut next = ab;
        for idx in 0..n {
            next[(idx, idx)] += coeff;
        }
        b = next;
    }
    let coeffs = clean_coefficients(coeffs);
    ensure_finite_coefficients("ss characteristic polynomial", &coeffs)?;
    Ok(coeffs)
}

fn state_space_numerator(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    c: &DMatrix<Complex64>,
    d: Complex64,
    denominator: &[Complex64],
) -> ConversionResult<Vec<Complex64>> {
    let degree = a.nrows();
    if degree == 0 {
        return Ok(vec![d]);
    }

    let mut points = Vec::with_capacity(degree + 1);
    let mut values = Vec::with_capacity(degree + 1);
    for point in interpolation_candidates(degree) {
        let Some(response) = state_space_response_at(a, b, c, d, point) else {
            continue;
        };
        points.push(point);
        values.push(response * poly_eval(denominator, point));
        if points.len() == degree + 1 {
            break;
        }
    }
    if points.len() != degree + 1 {
        return Err(conversion_error(
            "failed to find enough nonsingular interpolation points for ss model",
            ConversionErrorKind::Internal,
        ));
    }

    let mut vandermonde = DMatrix::<Complex64>::zeros(degree + 1, degree + 1);
    for (row, point) in points.iter().enumerate() {
        for col in 0..=degree {
            vandermonde[(row, col)] = point.powu((degree - col) as u32);
        }
    }
    let rhs = DVector::<Complex64>::from_vec(values);
    let coeffs = vandermonde.lu().solve(&rhs).ok_or_else(|| {
        conversion_error(
            "failed to solve ss numerator interpolation system",
            ConversionErrorKind::Internal,
        )
    })?;
    let coeffs = clean_coefficients(coeffs.iter().copied().collect());
    ensure_finite_coefficients("ss numerator polynomial", &coeffs)?;
    Ok(coeffs)
}

fn interpolation_candidates(degree: usize) -> Vec<Complex64> {
    let needed = degree + 1;
    let mut points = Vec::with_capacity(needed * 6);
    for radius_idx in 0..4 {
        let radius = 0.75 + radius_idx as f64;
        for idx in 0..needed {
            let angle = std::f64::consts::TAU * ((idx as f64 + 0.37) / needed as f64);
            points.push(Complex64::from_polar(radius, angle));
        }
    }
    for idx in 1..=(needed * 2) {
        let value = idx as f64;
        points.push(Complex64::new(value, 0.0));
        points.push(Complex64::new(-value, 0.0));
    }
    points
}

fn state_space_response_at(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    c: &DMatrix<Complex64>,
    d: Complex64,
    point: Complex64,
) -> Option<Complex64> {
    let n = a.nrows();
    if n == 0 {
        return Some(d);
    }
    let mut system = -a.clone();
    for idx in 0..n {
        system[(idx, idx)] += point;
    }
    let state = system.lu().solve(b)?;
    let output = c * state;
    Some(d + output[(0, 0)])
}
fn trim_leading_complex_zeros(coeffs: Vec<Complex64>) -> Vec<Complex64> {
    let first_nonzero = coeffs
        .iter()
        .position(|value| value.norm() > EPS)
        .unwrap_or(coeffs.len());
    coeffs[first_nonzero..].to_vec()
}

fn ensure_finite_coefficients(label: &str, coeffs: &[Complex64]) -> ConversionResult<()> {
    if coeffs
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(conversion_error(
            format!("{label} values must be finite"),
            ConversionErrorKind::InvalidModel,
        ));
    }
    Ok(())
}

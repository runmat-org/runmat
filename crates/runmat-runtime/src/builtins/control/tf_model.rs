//! Shared SISO transfer-function object parsing, construction, and algebra.

use std::cell::Cell;
use std::collections::HashMap;

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_builtins::{
    Access, CharArray, ClassDef, ComplexTensor, MethodDef, ObjectInstance, PropertyDef, Tensor,
    Value,
};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, dispatcher, BuiltinResult, RuntimeError};

pub const TF_CLASS: &str = "tf";
pub const SS_CLASS: &str = "ss";
pub const DEFAULT_CONTINUOUS_VARIABLE: &str = "s";
pub const DEFAULT_DISCRETE_VARIABLE: &str = "z";
pub const EPS: f64 = 1.0e-12;

thread_local! {
    static TF_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone, Debug)]
pub struct TfModel {
    pub numerator: Vec<Complex64>,
    pub denominator: Vec<Complex64>,
    pub variable: String,
    pub sample_time: f64,
    pub input_delay: f64,
    pub output_delay: f64,
}

#[derive(Clone, Debug)]
pub struct RealTfModel {
    pub numerator: Vec<f64>,
    pub denominator: Vec<f64>,
    pub sample_time: f64,
    pub input_delay: f64,
    pub output_delay: f64,
}

#[derive(Clone, Debug)]
pub struct TfOptions {
    pub variable: String,
    pub sample_time: f64,
}

impl Default for TfOptions {
    fn default() -> Self {
        Self {
            variable: DEFAULT_CONTINUOUS_VARIABLE.to_string(),
            sample_time: 0.0,
        }
    }
}

pub fn control_error(
    builtin: &'static str,
    identifier: &'static str,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(identifier)
        .build()
}

pub fn ensure_tf_class_registered() {
    TF_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        let mut properties = HashMap::new();
        for name in [
            "Numerator",
            "Denominator",
            "Variable",
            "Ts",
            "InputDelay",
            "OutputDelay",
        ] {
            properties.insert(
                name.to_string(),
                PropertyDef {
                    name: name.to_string(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: Access::Public,
                    set_access: Access::Public,
                    default_value: None,
                },
            );
        }

        let mut methods = HashMap::new();
        for method_name in [
            "plus", "minus", "uplus", "uminus", "times", "mtimes", "rdivide", "mrdivide",
            "ldivide", "mldivide", "power", "mpower",
        ] {
            methods.insert(
                method_name.to_string(),
                MethodDef {
                    name: method_name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: format!("{TF_CLASS}.{method_name}"),
                    implicit_class_argument: None,
                },
            );
        }

        runmat_builtins::register_class(ClassDef {
            name: TF_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
        registered.set(true);
    });
}

impl TfModel {
    pub fn new(
        numerator: Vec<Complex64>,
        denominator: Vec<Complex64>,
        options: TfOptions,
    ) -> BuiltinResult<Self> {
        Self::with_delays(numerator, denominator, options, 0.0, 0.0)
    }

    pub fn with_delays(
        numerator: Vec<Complex64>,
        denominator: Vec<Complex64>,
        options: TfOptions,
        input_delay: f64,
        output_delay: f64,
    ) -> BuiltinResult<Self> {
        validate_coefficients("numerator", &numerator, "tf")?;
        validate_coefficients("denominator", &denominator, "tf")?;
        if all_zero(&denominator) {
            return Err(control_error(
                "tf",
                "RunMat:tf:DenominatorInvalid",
                "tf: invalid denominator coefficients: denominator coefficients must not all be zero",
            ));
        }
        let variable = validate_variable(&options.variable, "tf")?;
        validate_sample_time(options.sample_time, "tf")?;
        validate_variable_domain(&variable, options.sample_time, "tf")?;
        validate_delay(input_delay, "InputDelay", "tf")?;
        validate_delay(output_delay, "OutputDelay", "tf")?;
        Ok(Self {
            numerator: trim_leading_complex_zeros(numerator),
            denominator: trim_leading_complex_zeros(denominator),
            variable,
            sample_time: options.sample_time,
            input_delay,
            output_delay,
        })
    }

    pub fn continuous_variable(variable: impl Into<String>) -> BuiltinResult<Self> {
        let variable = variable.into();
        Self::new(
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(1.0, 0.0)],
            TfOptions {
                variable,
                sample_time: 0.0,
            },
        )
    }

    pub fn discrete_variable(variable: impl Into<String>, sample_time: f64) -> BuiltinResult<Self> {
        Self::new(
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(1.0, 0.0)],
            TfOptions {
                variable: variable.into(),
                sample_time,
            },
        )
    }

    pub fn scalar(value: Complex64, options: TfOptions) -> BuiltinResult<Self> {
        Self::new(vec![value], vec![Complex64::new(1.0, 0.0)], options)
    }

    pub async fn from_value_async(value: Value, builtin: &'static str) -> BuiltinResult<Self> {
        let gathered = dispatcher::gather_if_needed_async(&value).await?;
        Self::from_value(gathered, builtin)
    }

    pub fn from_value(value: Value, builtin: &'static str) -> BuiltinResult<Self> {
        let Value::Object(object) = value else {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: expected a tf object"),
            ));
        };
        if !object.is_class(TF_CLASS) {
            return Err(control_error(
                builtin,
                unsupported_model_identifier(builtin),
                format!(
                    "{builtin}: unsupported model class '{}'; only SISO tf objects are supported",
                    object.class_name
                ),
            ));
        }

        let numerator = coefficients_from_property(&object, "Numerator", builtin)?;
        let denominator = coefficients_from_property(&object, "Denominator", builtin)?;
        let sample_time = scalar_property(&object, "Ts", builtin)?;
        let input_delay = scalar_property(&object, "InputDelay", builtin)?;
        let output_delay = scalar_property(&object, "OutputDelay", builtin)?;
        validate_sample_time(sample_time, builtin)?;
        validate_delay(input_delay, "InputDelay", builtin)?;
        validate_delay(output_delay, "OutputDelay", builtin)?;
        if all_zero(&denominator) {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: denominator coefficients must not all be zero"),
            ));
        }
        let variable = match object.properties.get("Variable") {
            Some(value) => validate_variable(&scalar_text(value, "Variable", builtin)?, builtin)?,
            None => {
                if sample_time > 0.0 {
                    DEFAULT_DISCRETE_VARIABLE.to_string()
                } else {
                    DEFAULT_CONTINUOUS_VARIABLE.to_string()
                }
            }
        };
        validate_variable_domain(&variable, sample_time, builtin)?;
        Ok(Self {
            numerator: trim_leading_complex_zeros(numerator),
            denominator: trim_leading_complex_zeros(denominator),
            variable,
            sample_time,
            input_delay,
            output_delay,
        })
    }

    pub fn to_value(&self, builtin: &'static str) -> BuiltinResult<Value> {
        ensure_tf_class_registered();
        let mut object = ObjectInstance::new(TF_CLASS.to_string());
        object.properties.insert(
            "Numerator".to_string(),
            coefficient_value(&self.numerator, builtin)?,
        );
        object.properties.insert(
            "Denominator".to_string(),
            coefficient_value(&self.denominator, builtin)?,
        );
        object.properties.insert(
            "Variable".to_string(),
            Value::CharArray(CharArray::new_row(&self.variable)),
        );
        object
            .properties
            .insert("Ts".to_string(), Value::Num(self.sample_time));
        object
            .properties
            .insert("InputDelay".to_string(), Value::Num(self.input_delay));
        object
            .properties
            .insert("OutputDelay".to_string(), Value::Num(self.output_delay));
        Ok(Value::Object(object))
    }

    pub fn to_real(&self, builtin: &'static str) -> BuiltinResult<RealTfModel> {
        let numerator = real_coefficients(&self.numerator, "Numerator", builtin)?;
        let denominator = real_coefficients(&self.denominator, "Denominator", builtin)?;
        Ok(RealTfModel {
            numerator,
            denominator,
            sample_time: self.sample_time,
            input_delay: self.input_delay,
            output_delay: self.output_delay,
        })
    }

    pub fn is_discrete(&self) -> bool {
        self.sample_time > 0.0
    }

    pub fn normalized(&self) -> BuiltinResult<Self> {
        let leading = *self.denominator.first().ok_or_else(|| {
            control_error(
                "tf",
                "RunMat:tf:DenominatorInvalid",
                "tf: denominator coefficients cannot be empty",
            )
        })?;
        if leading.norm() <= EPS {
            return Err(control_error(
                "tf",
                "RunMat:tf:DenominatorInvalid",
                "tf: leading denominator coefficient must be non-zero",
            ));
        }
        let mut out = self.clone();
        out.numerator = out.numerator.iter().map(|value| *value / leading).collect();
        out.denominator = out
            .denominator
            .iter()
            .map(|value| *value / leading)
            .collect();
        Ok(out)
    }

    pub fn add(&self, rhs: &Self) -> BuiltinResult<Self> {
        self.ensure_arithmetic_compatible(rhs, "plus")?;
        let numerator = poly_add(
            &poly_mul(&self.numerator, &rhs.denominator),
            &poly_mul(&rhs.numerator, &self.denominator),
        );
        let denominator = poly_mul(&self.denominator, &rhs.denominator);
        self.with_new_coefficients(numerator, denominator)
    }

    pub fn sub(&self, rhs: &Self) -> BuiltinResult<Self> {
        self.ensure_arithmetic_compatible(rhs, "minus")?;
        let numerator = poly_sub(
            &poly_mul(&self.numerator, &rhs.denominator),
            &poly_mul(&rhs.numerator, &self.denominator),
        );
        let denominator = poly_mul(&self.denominator, &rhs.denominator);
        self.with_new_coefficients(numerator, denominator)
    }

    pub fn neg(&self) -> BuiltinResult<Self> {
        self.with_new_coefficients(
            poly_scale(&self.numerator, -Complex64::new(1.0, 0.0)),
            self.denominator.clone(),
        )
    }

    pub fn mul(&self, rhs: &Self) -> BuiltinResult<Self> {
        self.ensure_arithmetic_compatible(rhs, "mtimes")?;
        self.with_new_coefficients(
            poly_mul(&self.numerator, &rhs.numerator),
            poly_mul(&self.denominator, &rhs.denominator),
        )
    }

    pub fn div(&self, rhs: &Self) -> BuiltinResult<Self> {
        self.ensure_arithmetic_compatible(rhs, "mrdivide")?;
        if all_zero(&rhs.numerator) {
            return Err(control_error(
                "tf",
                "RunMat:tf:DivideByZero",
                "tf: cannot divide by a zero transfer function",
            ));
        }
        self.with_new_coefficients(
            poly_mul(&self.numerator, &rhs.denominator),
            poly_mul(&self.denominator, &rhs.numerator),
        )
    }

    pub fn powi(&self, exponent: i64) -> BuiltinResult<Self> {
        let one = Self::scalar(
            Complex64::new(1.0, 0.0),
            TfOptions {
                variable: self.variable.clone(),
                sample_time: self.sample_time,
            },
        )?;
        if exponent == 0 {
            return Ok(one);
        }
        let mut base = self.clone();
        let mut exp = exponent;
        if exp < 0 {
            base = one.div(&base)?;
            exp = exp.checked_neg().ok_or_else(|| {
                control_error(
                    "tf",
                    "RunMat:tf:InvalidExponent",
                    "tf: exponent magnitude is too large",
                )
            })?;
        }
        let mut result = one;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base)?;
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    pub fn poles(&self) -> BuiltinResult<Vec<Complex64>> {
        polynomial_roots(&self.denominator, "pole")
    }

    pub fn zeros(&self) -> BuiltinResult<Vec<Complex64>> {
        polynomial_roots(&self.numerator, "zero")
    }

    pub fn dc_gain(&self) -> BuiltinResult<Complex64> {
        let point = if self.is_discrete() {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
        let num = poly_eval(&self.numerator, point);
        let den = poly_eval(&self.denominator, point);
        if den.norm() <= EPS {
            if num.norm() <= EPS {
                Ok(Complex64::new(f64::NAN, f64::NAN))
            } else {
                // Direct division by a zero complex denominator produces a spurious NaN
                // component. Factor the pole so the infinite result retains the local
                // signed/complex-axis direction of the positive-real approach to the DC point.
                let local_denominator =
                    first_nonzero_local_polynomial_coefficient(&self.denominator, point);
                Ok(complex_infinity_in_direction(num / local_denominator))
            }
        } else {
            Ok(num / den)
        }
    }

    pub fn is_stable(&self) -> BuiltinResult<bool> {
        let poles = self.poles()?;
        if self.is_discrete() {
            Ok(poles.iter().all(|pole| pole.norm() < 1.0 - EPS))
        } else {
            Ok(poles.iter().all(|pole| pole.re < -EPS))
        }
    }

    pub fn feedback(&self, rhs: &Self, sign: f64) -> BuiltinResult<Self> {
        if sign != -1.0 && sign != 1.0 {
            return Err(control_error(
                "feedback",
                "RunMat:feedback:InvalidSign",
                "feedback: sign must be -1 or +1",
            ));
        }
        self.ensure_arithmetic_compatible(rhs, "feedback")?;
        let numerator = poly_mul(&self.numerator, &rhs.denominator);
        let base_denominator = poly_mul(&self.denominator, &rhs.denominator);
        let loop_numerator = poly_mul(&self.numerator, &rhs.numerator);
        let denominator = if sign < 0.0 {
            poly_add(&base_denominator, &loop_numerator)
        } else {
            poly_sub(&base_denominator, &loop_numerator)
        };
        self.with_new_coefficients(numerator, denominator)
    }

    fn ensure_arithmetic_compatible(&self, rhs: &Self, op: &'static str) -> BuiltinResult<()> {
        if (self.sample_time - rhs.sample_time).abs() > EPS {
            return Err(control_error(
                "tf",
                "RunMat:tf:SampleTimeMismatch",
                format!("tf.{op}: sample times must match"),
            ));
        }
        if self.variable != rhs.variable {
            return Err(control_error(
                "tf",
                "RunMat:tf:VariableMismatch",
                format!("tf.{op}: transfer-function variables must match"),
            ));
        }
        if self.input_delay.abs() > EPS
            || self.output_delay.abs() > EPS
            || rhs.input_delay.abs() > EPS
            || rhs.output_delay.abs() > EPS
        {
            return Err(control_error(
                "tf",
                "RunMat:tf:UnsupportedDelay",
                format!("tf.{op}: input and output delays are not supported in arithmetic"),
            ));
        }
        Ok(())
    }

    fn with_new_coefficients(
        &self,
        numerator: Vec<Complex64>,
        denominator: Vec<Complex64>,
    ) -> BuiltinResult<Self> {
        Self::with_delays(
            numerator,
            denominator,
            TfOptions {
                variable: self.variable.clone(),
                sample_time: self.sample_time,
            },
            self.input_delay,
            self.output_delay,
        )
    }
}

impl RealTfModel {
    pub fn normalized(&self, builtin: &'static str) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
        let den = trim_leading_real_zeros(self.denominator.clone());
        if den.is_empty() || den[0].abs() <= EPS {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: leading denominator coefficient must be non-zero"),
            ));
        }
        let num = trim_leading_real_zeros(self.numerator.clone());
        let leading = den[0];
        Ok((
            num.iter().map(|value| value / leading).collect(),
            den.iter().map(|value| value / leading).collect(),
        ))
    }

    pub fn ensure_zero_delays(&self, builtin: &'static str) -> BuiltinResult<()> {
        if self.input_delay.abs() > EPS || self.output_delay.abs() > EPS {
            return Err(control_error(
                builtin,
                unsupported_model_identifier(builtin),
                format!(
                    "{builtin}: transfer functions with input or output delays are not supported"
                ),
            ));
        }
        Ok(())
    }
}

pub async fn parse_coefficients(
    label: &str,
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<Vec<Complex64>> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    let coeffs = match gathered {
        Value::Tensor(tensor) => {
            ensure_vector_shape(label, &tensor.shape, builtin)?;
            tensor::tensor_values_f64(&tensor)
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect()
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(label, &tensor.shape, builtin)?;
            tensor::complex_tensor_into_values_complex64(tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|err| {
                control_error(
                    builtin,
                    invalid_coefficients_identifier(builtin),
                    format!("{builtin}: failed to convert logical array: {err}"),
                )
            })?;
            ensure_vector_shape(label, &tensor.shape, builtin)?;
            tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect()
        }
        Value::Num(n) => vec![Complex64::new(n, 0.0)],
        Value::Int(i) => vec![Complex64::new(i.to_f64(), 0.0)],
        Value::Bool(b) => vec![Complex64::new(if b { 1.0 } else { 0.0 }, 0.0)],
        Value::Complex(re, im) => vec![Complex64::new(re, im)],
        other => {
            return Err(control_error(
                builtin,
                invalid_coefficients_identifier(builtin),
                format!("{builtin}: {label} must be a numeric coefficient vector, got {other:?}"),
            ));
        }
    };
    validate_coefficients(label, &coeffs, builtin)?;
    Ok(coeffs)
}

pub async fn value_to_model_with_reference(
    value: Value,
    reference: &TfModel,
    builtin: &'static str,
) -> BuiltinResult<TfModel> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    if matches!(gathered, Value::Object(_)) {
        return TfModel::from_value(gathered, builtin);
    }
    let scalar = scalar_complex(&gathered, builtin)?;
    TfModel::scalar(
        scalar,
        TfOptions {
            variable: reference.variable.clone(),
            sample_time: reference.sample_time,
        },
    )
}

pub async fn two_models_ordered(
    lhs: Value,
    rhs: Value,
    builtin: &'static str,
) -> BuiltinResult<(TfModel, TfModel)> {
    let lhs = dispatcher::gather_if_needed_async(&lhs).await?;
    let rhs = dispatcher::gather_if_needed_async(&rhs).await?;
    match (lhs, rhs) {
        (left @ Value::Object(_), right @ Value::Object(_)) => Ok((
            TfModel::from_value(left, builtin)?,
            TfModel::from_value(right, builtin)?,
        )),
        (left @ Value::Object(_), right) => {
            let left_model = TfModel::from_value(left, builtin)?;
            let right_model = value_to_model_with_reference(right, &left_model, builtin).await?;
            Ok((left_model, right_model))
        }
        (left, right @ Value::Object(_)) => {
            let right_model = TfModel::from_value(right, builtin)?;
            let left_model = value_to_model_with_reference(left, &right_model, builtin).await?;
            Ok((left_model, right_model))
        }
        (left, right) => {
            let options = TfOptions::default();
            Ok((
                TfModel::scalar(scalar_complex(&left, builtin)?, options.clone())?,
                TfModel::scalar(scalar_complex(&right, builtin)?, options)?,
            ))
        }
    }
}

pub fn scalar_text(value: &Value, context: &str, builtin: &'static str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        other => Err(control_error(
            builtin,
            invalid_argument_identifier(builtin),
            format!(
                "{builtin}: {context} must be a string scalar or character vector, got {other:?}"
            ),
        )),
    }
}

pub fn scalar_f64(value: &Value, context: &str, builtin: &'static str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] == 0 { 0.0 } else { 1.0 })
        }
        other => Err(control_error(
            builtin,
            invalid_argument_identifier(builtin),
            format!("{builtin}: {context} must be a real scalar, got {other:?}"),
        )),
    }
}

pub fn scalar_complex(value: &Value, builtin: &'static str) -> BuiltinResult<Complex64> {
    match value {
        Value::Num(n) => Ok(Complex64::new(*n, 0.0)),
        Value::Int(i) => Ok(Complex64::new(i.to_f64(), 0.0)),
        Value::Bool(b) => Ok(Complex64::new(if *b { 1.0 } else { 0.0 }, 0.0)),
        Value::Complex(re, im) => Ok(Complex64::new(*re, *im)),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(Complex64::new(tensor::tensor_value_f64(tensor, 0), 0.0))
        }
        Value::ComplexTensor(tensor) if tensor::is_scalar_complex_tensor(tensor) => {
            Ok(tensor::complex_tensor_value_complex64(tensor, 0))
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => Ok(Complex64::new(
            if logical.data[0] == 0 { 0.0 } else { 1.0 },
            0.0,
        )),
        other => Err(control_error(
            builtin,
            invalid_argument_identifier(builtin),
            format!("{builtin}: expected a scalar numeric value or tf object, got {other:?}"),
        )),
    }
}

pub fn validate_variable(variable: &str, builtin: &'static str) -> BuiltinResult<String> {
    let variable = variable.trim();
    match variable {
        "s" | "p" | "z" | "q" | "z^-1" | "q^-1" => Ok(variable.to_string()),
        _ => Err(control_error(
            builtin,
            "RunMat:tf:InvalidVariable",
            "tf: invalid Variable option: must be one of 's', 'p', 'z', 'q', 'z^-1', or 'q^-1'",
        )),
    }
}

pub fn is_discrete_variable(variable: &str) -> bool {
    matches!(variable.trim(), "z" | "q" | "z^-1" | "q^-1")
}

pub fn validate_variable_domain(
    variable: &str,
    sample_time: f64,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let discrete_variable = is_discrete_variable(variable);
    if discrete_variable && sample_time != -1.0 && sample_time <= 0.0 {
        return Err(control_error(
            builtin,
            "RunMat:tf:InvalidSampleTime",
            format!(
                "{builtin}: discrete transfer-function variables require a positive sample time"
            ),
        ));
    }
    if !discrete_variable && sample_time != 0.0 {
        return Err(control_error(
            builtin,
            "RunMat:tf:InvalidVariable",
            format!("{builtin}: continuous transfer-function variables require Ts = 0"),
        ));
    }
    Ok(())
}

pub fn validate_sample_time(sample_time: f64, builtin: &'static str) -> BuiltinResult<()> {
    if !sample_time.is_finite() || (sample_time < 0.0 && sample_time != -1.0) {
        return Err(control_error(
            builtin,
            invalid_sample_time_identifier(builtin),
            format!("{builtin}: sample time must be -1 or a finite non-negative scalar"),
        ));
    }
    Ok(())
}

pub fn output_complex_scalar(value: Complex64) -> Value {
    if value.im.abs() <= EPS {
        Value::Num(value.re)
    } else {
        Value::Complex(value.re, value.im)
    }
}

pub fn output_complex_column(
    values: Vec<Complex64>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let rows = values.len();
    if values.iter().all(|value| value.im.abs() <= EPS) {
        let data = values.into_iter().map(|value| value.re).collect::<Vec<_>>();
        Ok(Value::Tensor(Tensor::new(data, vec![rows, 1]).map_err(
            |err| {
                control_error(
                    builtin,
                    internal_identifier(builtin),
                    format!("{builtin}: failed to build output tensor: {err}"),
                )
            },
        )?))
    } else {
        let data = values
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect::<Vec<_>>();
        Ok(Value::ComplexTensor(
            ComplexTensor::new(data, vec![rows, 1]).map_err(|err| {
                control_error(
                    builtin,
                    internal_identifier(builtin),
                    format!("{builtin}: failed to build complex output tensor: {err}"),
                )
            })?,
        ))
    }
}

pub fn ss_poles_from_object(
    object: &ObjectInstance,
    builtin: &'static str,
) -> BuiltinResult<(Vec<Complex64>, f64)> {
    let a = ss_state_matrix_property(object, "A", builtin)?;
    let sample_time = scalar_property(object, "Ts", builtin)?;
    validate_sample_time(sample_time, builtin)?;
    let eigenvalues = a.eigenvalues().ok_or_else(|| {
        control_error(
            builtin,
            internal_identifier(builtin),
            format!("{builtin}: failed to compute state matrix eigenvalues"),
        )
    })?;
    Ok((eigenvalues.iter().copied().collect(), sample_time))
}

fn ss_state_matrix_property(
    object: &ObjectInstance,
    name: &'static str,
    builtin: &'static str,
) -> BuiltinResult<DMatrix<Complex64>> {
    let value = object.properties.get(name).ok_or_else(|| {
        control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: ss object is missing {name}"),
        )
    })?;
    let tensor = match value {
        Value::Tensor(tensor) => tensor::integer_tensor_to_f64(tensor.clone()).map_err(|err| {
            control_error(
                builtin,
                internal_identifier(builtin),
                format!("{builtin}: failed to normalize ss {name}: {err}"),
            )
        })?,
        Value::Num(n) => Tensor::new(vec![*n], vec![1, 1]).map_err(|err| {
            control_error(
                builtin,
                internal_identifier(builtin),
                format!("{builtin}: failed to build scalar matrix: {err}"),
            )
        })?,
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|err| {
            control_error(
                builtin,
                internal_identifier(builtin),
                format!("{builtin}: failed to build scalar matrix: {err}"),
            )
        })?,
        other => {
            return Err(control_error(
                builtin,
                unsupported_model_identifier(builtin),
                format!("{builtin}: ss {name} must be a finite real matrix, got {other:?}"),
            ));
        }
    };
    if tensor.shape.len() > 2 || tensor.rows != tensor.cols {
        return Err(control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!(
                "{builtin}: ss {name} must be square, got {:?}",
                tensor.shape
            ),
        ));
    }
    let values = tensor::tensor_values_f64_cow(&tensor);
    if values.iter().any(|value| !value.is_finite()) {
        return Err(control_error(
            builtin,
            unsupported_model_identifier(builtin),
            format!("{builtin}: ss {name} must contain only finite real values"),
        ));
    }
    let mut matrix = DMatrix::<Complex64>::zeros(tensor.rows, tensor.cols);
    for col in 0..tensor.cols {
        for row in 0..tensor.rows {
            matrix[(row, col)] = Complex64::new(values[row + col * tensor.rows], 0.0);
        }
    }
    Ok(matrix)
}

pub fn polynomial_roots(
    coeffs: &[Complex64],
    builtin: &'static str,
) -> BuiltinResult<Vec<Complex64>> {
    let trimmed = trim_leading_complex_zeros(coeffs.to_vec());
    if trimmed.len() <= 1 {
        return Ok(Vec::new());
    }
    if trimmed.len() == 2 {
        return Ok(vec![-trimmed[1] / trimmed[0]]);
    }
    let degree = trimmed.len() - 1;
    let leading = trimmed[0];
    if leading.norm() <= EPS {
        return Err(control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: leading polynomial coefficient must be non-zero"),
        ));
    }
    let mut companion = DMatrix::<Complex64>::zeros(degree, degree);
    for row in 1..degree {
        companion[(row, row - 1)] = Complex64::new(1.0, 0.0);
    }
    for (idx, coeff) in trimmed.iter().enumerate().skip(1) {
        companion[(0, idx - 1)] = -*coeff / leading;
    }
    let eigenvalues = companion.eigenvalues().ok_or_else(|| {
        control_error(
            builtin,
            internal_identifier(builtin),
            format!("{builtin}: failed to compute polynomial roots"),
        )
    })?;
    Ok(eigenvalues.iter().copied().collect())
}

pub fn poly_eval(coeffs: &[Complex64], x: Complex64) -> Complex64 {
    coeffs
        .iter()
        .fold(Complex64::new(0.0, 0.0), |acc, coeff| acc * x + *coeff)
}

fn first_nonzero_local_polynomial_coefficient(coeffs: &[Complex64], point: Complex64) -> Complex64 {
    // Synthetic division by (x - point) removes one zero at the evaluation point per pass.
    let mut quotient = coeffs.to_vec();
    while quotient.len() > 1 && poly_eval(&quotient, point).norm() <= EPS {
        let mut reduced = Vec::with_capacity(quotient.len() - 1);
        let mut synthetic = quotient[0];
        reduced.push(synthetic);
        for coefficient in quotient.iter().take(quotient.len() - 1).skip(1) {
            synthetic = *coefficient + point * synthetic;
            reduced.push(synthetic);
        }
        quotient = reduced;
    }
    poly_eval(&quotient, point)
}

fn complex_infinity_in_direction(direction: Complex64) -> Complex64 {
    let scale = direction.re.abs().max(direction.im.abs());
    if scale.is_nan() {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    if scale.is_infinite() {
        let component = |value: f64| {
            if value.is_nan() {
                f64::NAN
            } else if value.is_infinite() {
                f64::INFINITY.copysign(value)
            } else {
                0.0
            }
        };
        return Complex64::new(component(direction.re), component(direction.im));
    }
    if scale == 0.0 {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    let component = |value: f64| {
        if value.abs() <= EPS * scale {
            0.0
        } else {
            f64::INFINITY.copysign(value)
        }
    };
    Complex64::new(component(direction.re), component(direction.im))
}

pub fn trim_leading_real_zeros(coeffs: Vec<f64>) -> Vec<f64> {
    let first_nonzero = coeffs
        .iter()
        .position(|value| value.abs() > EPS)
        .unwrap_or(coeffs.len());
    if first_nonzero == coeffs.len() {
        return vec![0.0];
    }
    coeffs[first_nonzero..].to_vec()
}

fn coefficients_from_property(
    object: &ObjectInstance,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<Vec<Complex64>> {
    let value = object.properties.get(name).ok_or_else(|| {
        control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: tf object is missing {name}"),
        )
    })?;
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(name, &tensor.shape, builtin)?;
            Ok(tensor::tensor_values_f64(tensor)
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect())
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(name, &tensor.shape, builtin)?;
            Ok(tensor::complex_tensor_values_complex64(tensor))
        }
        Value::Num(n) => Ok(vec![Complex64::new(*n, 0.0)]),
        Value::Int(i) => Ok(vec![Complex64::new(i.to_f64(), 0.0)]),
        Value::Bool(b) => Ok(vec![Complex64::new(if *b { 1.0 } else { 0.0 }, 0.0)]),
        other => Err(control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: tf {name} coefficients must be numeric, got {other:?}"),
        )),
    }
}

fn coefficient_value(coeffs: &[Complex64], builtin: &'static str) -> BuiltinResult<Value> {
    let len = coeffs.len();
    if coeffs.iter().all(|coeff| coeff.im.abs() <= EPS) {
        let data = coeffs.iter().map(|coeff| coeff.re).collect::<Vec<_>>();
        let tensor = Tensor::new(data, vec![1, len]).map_err(|err| {
            control_error(
                builtin,
                internal_identifier(builtin),
                format!("{builtin}: failed to build coefficient tensor: {err}"),
            )
        })?;
        Ok(Value::Tensor(tensor))
    } else {
        let data = coeffs
            .iter()
            .map(|coeff| (coeff.re, coeff.im))
            .collect::<Vec<_>>();
        let tensor = ComplexTensor::new(data, vec![1, len]).map_err(|err| {
            control_error(
                builtin,
                internal_identifier(builtin),
                format!("{builtin}: failed to build complex coefficient tensor: {err}"),
            )
        })?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn property<'a>(
    object: &'a ObjectInstance,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<&'a Value> {
    object.properties.get(name).ok_or_else(|| {
        control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: tf object is missing {name} property"),
        )
    })
}

fn scalar_property(
    object: &ObjectInstance,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<f64> {
    scalar_f64(property(object, name, builtin)?, name, builtin)
}

fn validate_delay(value: f64, label: &str, builtin: &'static str) -> BuiltinResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: {label} must be a finite non-negative scalar"),
        ));
    }
    Ok(())
}

fn validate_coefficients(
    label: &str,
    coeffs: &[Complex64],
    builtin: &'static str,
) -> BuiltinResult<()> {
    if coeffs.is_empty() {
        return Err(control_error(
            builtin,
            invalid_coefficients_identifier(builtin),
            format!("{builtin}: {label} coefficients cannot be empty"),
        ));
    }
    for coeff in coeffs {
        if !coeff.re.is_finite() || !coeff.im.is_finite() {
            return Err(control_error(
                builtin,
                invalid_coefficients_identifier(builtin),
                format!("{builtin}: {label} coefficients must be finite"),
            ));
        }
    }
    Ok(())
}

fn real_coefficients(
    coeffs: &[Complex64],
    label: &str,
    builtin: &'static str,
) -> BuiltinResult<Vec<f64>> {
    let mut out = Vec::with_capacity(coeffs.len());
    for coeff in coeffs {
        if coeff.im.abs() > EPS {
            return Err(control_error(
                builtin,
                unsupported_model_identifier(builtin),
                format!("{builtin}: complex tf {label} coefficients are not supported"),
            ));
        }
        out.push(coeff.re);
    }
    Ok(out)
}

fn ensure_vector_shape(label: &str, shape: &[usize], builtin: &'static str) -> BuiltinResult<()> {
    let non_unit = shape.iter().copied().filter(|&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(control_error(
            builtin,
            invalid_coefficients_identifier(builtin),
            format!("{builtin}: {label} coefficients must be a vector"),
        ))
    }
}

fn trim_leading_complex_zeros(coeffs: Vec<Complex64>) -> Vec<Complex64> {
    let first_nonzero = coeffs
        .iter()
        .position(|value| value.norm() > EPS)
        .unwrap_or(coeffs.len());
    let trimmed = coeffs[first_nonzero..].to_vec();
    if trimmed.is_empty() {
        vec![Complex64::new(0.0, 0.0)]
    } else {
        trimmed
    }
}

fn all_zero(coeffs: &[Complex64]) -> bool {
    coeffs.iter().all(|coeff| coeff.norm() <= EPS)
}

fn poly_add(lhs: &[Complex64], rhs: &[Complex64]) -> Vec<Complex64> {
    let len = lhs.len().max(rhs.len());
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    for (idx, value) in lhs.iter().enumerate() {
        out[len - lhs.len() + idx] += *value;
    }
    for (idx, value) in rhs.iter().enumerate() {
        out[len - rhs.len() + idx] += *value;
    }
    trim_leading_complex_zeros(out)
}

fn poly_sub(lhs: &[Complex64], rhs: &[Complex64]) -> Vec<Complex64> {
    poly_add(lhs, &poly_scale(rhs, -Complex64::new(1.0, 0.0)))
}

fn poly_scale(coeffs: &[Complex64], scale: Complex64) -> Vec<Complex64> {
    trim_leading_complex_zeros(coeffs.iter().map(|value| *value * scale).collect())
}

fn poly_mul(lhs: &[Complex64], rhs: &[Complex64]) -> Vec<Complex64> {
    if all_zero(lhs) || all_zero(rhs) {
        return vec![Complex64::new(0.0, 0.0)];
    }
    let mut out = vec![Complex64::new(0.0, 0.0); lhs.len() + rhs.len() - 1];
    for (i, a) in lhs.iter().enumerate() {
        for (j, b) in rhs.iter().enumerate() {
            out[i + j] += *a * *b;
        }
    }
    trim_leading_complex_zeros(out)
}

fn invalid_argument_identifier(builtin: &str) -> &'static str {
    match builtin {
        "feedback" => "RunMat:feedback:InvalidArgument",
        "stepinfo" => "RunMat:stepinfo:InvalidArgument",
        "dcgain" => "RunMat:dcgain:InvalidArgument",
        "pole" => "RunMat:pole:InvalidArgument",
        "zero" => "RunMat:zero:InvalidArgument",
        "damp" => "RunMat:damp:InvalidModel",
        "rlocus" => "RunMat:rlocus:InvalidArgument",
        "pzmap" => "RunMat:pzmap:InvalidArgument",
        "isstable" => "RunMat:isstable:InvalidArgument",
        _ => "RunMat:tf:InvalidArgument",
    }
}

fn invalid_coefficients_identifier(builtin: &str) -> &'static str {
    match builtin {
        "feedback" => "RunMat:feedback:InvalidModel",
        "stepinfo" => "RunMat:stepinfo:InvalidData",
        "dcgain" => "RunMat:dcgain:InvalidModel",
        "pole" => "RunMat:pole:InvalidModel",
        "zero" => "RunMat:zero:InvalidModel",
        "damp" => "RunMat:damp:InvalidModel",
        "rlocus" => "RunMat:rlocus:InvalidModel",
        "pzmap" => "RunMat:pzmap:InvalidModel",
        "isstable" => "RunMat:isstable:InvalidModel",
        _ => "RunMat:tf:InvalidCoefficients",
    }
}

fn invalid_sample_time_identifier(builtin: &str) -> &'static str {
    match builtin {
        "feedback" => "RunMat:feedback:InvalidSampleTime",
        "stepinfo" => "RunMat:stepinfo:InvalidArgument",
        "dcgain" => "RunMat:dcgain:InvalidModel",
        "pole" => "RunMat:pole:InvalidModel",
        "zero" => "RunMat:zero:InvalidModel",
        "damp" => "RunMat:damp:InvalidModel",
        "rlocus" => "RunMat:rlocus:InvalidModel",
        "pzmap" => "RunMat:pzmap:InvalidModel",
        "isstable" => "RunMat:isstable:InvalidModel",
        _ => "RunMat:tf:InvalidSampleTime",
    }
}

fn invalid_model_identifier(builtin: &str) -> &'static str {
    match builtin {
        "feedback" => "RunMat:feedback:InvalidModel",
        "stepinfo" => "RunMat:stepinfo:InvalidSystem",
        "dcgain" => "RunMat:dcgain:InvalidModel",
        "pole" => "RunMat:pole:InvalidModel",
        "zero" => "RunMat:zero:InvalidModel",
        "damp" => "RunMat:damp:InvalidModel",
        "rlocus" => "RunMat:rlocus:InvalidModel",
        "pzmap" => "RunMat:pzmap:InvalidModel",
        "isstable" => "RunMat:isstable:InvalidModel",
        _ => "RunMat:tf:InvalidModel",
    }
}

fn unsupported_model_identifier(builtin: &str) -> &'static str {
    match builtin {
        "feedback" => "RunMat:feedback:UnsupportedModel",
        "stepinfo" => "RunMat:stepinfo:UnsupportedModel",
        "dcgain" => "RunMat:dcgain:UnsupportedModel",
        "pole" => "RunMat:pole:UnsupportedModel",
        "zero" => "RunMat:zero:UnsupportedModel",
        "damp" => "RunMat:damp:UnsupportedModel",
        "rlocus" => "RunMat:rlocus:UnsupportedModel",
        "pzmap" => "RunMat:pzmap:UnsupportedModel",
        "isstable" => "RunMat:isstable:UnsupportedModel",
        _ => "RunMat:tf:UnsupportedModel",
    }
}

fn internal_identifier(builtin: &str) -> &'static str {
    match builtin {
        "feedback" => "RunMat:feedback:Internal",
        "stepinfo" => "RunMat:stepinfo:Internal",
        "dcgain" => "RunMat:dcgain:Internal",
        "pole" => "RunMat:pole:Internal",
        "zero" => "RunMat:zero:Internal",
        "damp" => "RunMat:damp:Internal",
        "rlocus" => "RunMat:rlocus:Internal",
        "pzmap" => "RunMat:pzmap:Internal",
        "isstable" => "RunMat:isstable:Internal",
        _ => "RunMat:tf:Internal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage};

    fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let is_scalar = shape.iter().product::<usize>() == 1;
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        if is_scalar {
        } else {
        }
        Value::Tensor(tensor)
    }

    fn poisoned_complex_integer_tensor(
        real: IntegerStorage,
        imag: IntegerStorage,
        shape: Vec<usize>,
    ) -> Value {
        let is_scalar = shape.iter().product::<usize>() == 1;
        let storage = IntegerComplexStorage::new(real, imag).expect("complex integer storage");
        let tensor = ComplexTensor::new_integer(storage, shape).expect("complex integer tensor");
        if is_scalar {
        } else {
        }
        Value::ComplexTensor(tensor)
    }

    #[test]
    fn dc_gain_returns_signed_infinity_for_continuous_integrators() {
        for (numerator, denominator, expected_negative) in [
            (
                vec![Complex64::new(2.0, 0.0)],
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                false,
            ),
            (
                vec![Complex64::new(-2.0, 0.0)],
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                true,
            ),
            (
                vec![Complex64::new(2.0, 0.0)],
                vec![Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
                true,
            ),
            (
                vec![Complex64::new(2.0, 0.0)],
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
                false,
            ),
        ] {
            let model = TfModel::new(numerator, denominator, TfOptions::default()).unwrap();
            let gain = model.dc_gain().unwrap();
            assert!(gain.re.is_infinite());
            assert_eq!(gain.re.is_sign_negative(), expected_negative);
            assert_eq!(gain.im, 0.0);
        }
    }

    #[test]
    fn dc_gain_preserves_complex_axis_and_handles_discrete_integrators() {
        let complex_model = TfModel::new(
            vec![Complex64::new(0.0, -3.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            TfOptions::default(),
        )
        .unwrap();
        let complex_gain = complex_model.dc_gain().unwrap();
        assert_eq!(complex_gain.re, 0.0);
        assert!(complex_gain.im.is_infinite() && complex_gain.im.is_sign_negative());

        let discrete_model = TfModel::new(
            vec![Complex64::new(4.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)],
            TfOptions {
                variable: DEFAULT_DISCRETE_VARIABLE.to_string(),
                sample_time: 0.25,
            },
        )
        .unwrap();
        let discrete_gain = discrete_model.dc_gain().unwrap();
        assert!(discrete_gain.re.is_infinite() && discrete_gain.re.is_sign_positive());
        assert_eq!(discrete_gain.im, 0.0);
    }

    #[test]
    fn complex_infinity_direction_survives_overflowed_components() {
        let finite_overflow = complex_infinity_in_direction(Complex64::new(f64::MAX, -f64::MAX));
        assert!(finite_overflow.re.is_infinite() && finite_overflow.re.is_sign_positive());
        assert!(finite_overflow.im.is_infinite() && finite_overflow.im.is_sign_negative());

        let nonfinite =
            complex_infinity_in_direction(Complex64::new(f64::INFINITY, -f64::INFINITY));
        assert!(nonfinite.re.is_infinite() && nonfinite.re.is_sign_positive());
        assert!(nonfinite.im.is_infinite() && nonfinite.im.is_sign_negative());

        let axis = complex_infinity_in_direction(Complex64::new(f64::INFINITY, 1.0));
        assert!(axis.re.is_infinite() && axis.re.is_sign_positive());
        assert_eq!(axis.im, 0.0);
    }

    #[test]
    fn dc_gain_keeps_exact_evaluation_point_cancellation_indeterminate() {
        let model = TfModel::new(
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            TfOptions::default(),
        )
        .unwrap();
        let gain = model.dc_gain().unwrap();
        assert!(gain.re.is_nan() && gain.im.is_nan());
    }

    #[test]
    fn scalar_f64_reads_typed_integer_storage_exactly() {
        let value = poisoned_integer_tensor(IntegerStorage::I16(vec![-7]), vec![1, 1]);
        assert_eq!(scalar_f64(&value, "Ts", "tf").expect("scalar"), -7.0);
    }

    #[test]
    fn scalar_complex_reads_typed_integer_storage_exactly() {
        let value = poisoned_integer_tensor(IntegerStorage::U64(vec![42]), vec![1, 1]);
        assert_eq!(
            scalar_complex(&value, "tf").expect("scalar"),
            Complex64::new(42.0, 0.0)
        );
    }

    #[test]
    fn scalar_complex_reads_complex_typed_integer_storage_exactly() {
        let value = poisoned_complex_integer_tensor(
            IntegerStorage::I16(vec![3]),
            IntegerStorage::I16(vec![-4]),
            vec![1, 1],
        );
        assert_eq!(
            scalar_complex(&value, "tf").expect("scalar"),
            Complex64::new(3.0, -4.0)
        );
    }

    #[test]
    fn parse_coefficients_reads_complex_typed_integer_storage_exactly() {
        let value = poisoned_complex_integer_tensor(
            IntegerStorage::I16(vec![1, 3]),
            IntegerStorage::I16(vec![2, -4]),
            vec![1, 2],
        );
        assert_eq!(
            block_on(parse_coefficients("numerator", value, "tf")).expect("coefficients"),
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)]
        );
    }

    #[test]
    fn coefficients_from_property_reads_complex_typed_integer_storage_exactly() {
        let mut object = ObjectInstance::new(TF_CLASS.to_string());
        object.properties.insert(
            "Numerator".to_string(),
            poisoned_complex_integer_tensor(
                IntegerStorage::I16(vec![5, 7]),
                IntegerStorage::I16(vec![-6, 8]),
                vec![1, 2],
            ),
        );

        assert_eq!(
            coefficients_from_property(&object, "Numerator", "tf").expect("coefficients"),
            vec![Complex64::new(5.0, -6.0), Complex64::new(7.0, 8.0)]
        );
    }

    #[test]
    fn ss_state_matrix_property_reads_typed_integer_storage_exactly() {
        let mut object = ObjectInstance::new(SS_CLASS.to_string());
        object.properties.insert(
            "A".to_string(),
            poisoned_integer_tensor(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2]),
        );

        let matrix = ss_state_matrix_property(&object, "A", "pole").expect("state matrix");
        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix[(0, 0)], Complex64::new(1.0, 0.0));
        assert_eq!(matrix[(1, 0)], Complex64::new(3.0, 0.0));
        assert_eq!(matrix[(0, 1)], Complex64::new(2.0, 0.0));
        assert_eq!(matrix[(1, 1)], Complex64::new(4.0, 0.0));
    }
}

//! Shared SISO transfer-function object parsing, construction, and algebra.

use std::collections::HashMap;
use std::sync::OnceLock;

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

static TF_CLASS_REGISTERED: OnceLock<()> = OnceLock::new();

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
    TF_CLASS_REGISTERED.get_or_init(|| {
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
                Ok(num / Complex64::new(0.0, 0.0))
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
            tensor
                .data
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect()
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(label, &tensor.shape, builtin)?;
            tensor
                .data
                .into_iter()
                .map(|(re, im)| Complex64::new(re, im))
                .collect()
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
            tensor
                .data
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
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
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
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(Complex64::new(tensor.data[0], 0.0)),
        Value::ComplexTensor(tensor) if tensor.data.len() == 1 => {
            let (re, im) = tensor.data[0];
            Ok(Complex64::new(re, im))
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
    if discrete_variable && sample_time <= 0.0 {
        return Err(control_error(
            builtin,
            "RunMat:tf:InvalidSampleTime",
            format!(
                "{builtin}: discrete transfer-function variables require a positive sample time"
            ),
        ));
    }
    if !discrete_variable && sample_time > 0.0 {
        return Err(control_error(
            builtin,
            "RunMat:tf:InvalidVariable",
            format!("{builtin}: continuous transfer-function variables require Ts = 0"),
        ));
    }
    Ok(())
}

pub fn validate_sample_time(sample_time: f64, builtin: &'static str) -> BuiltinResult<()> {
    if !sample_time.is_finite() || sample_time < 0.0 {
        return Err(control_error(
            builtin,
            invalid_sample_time_identifier(builtin),
            format!("{builtin}: sample time must be a finite non-negative scalar"),
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
        Value::Tensor(tensor) => tensor.clone(),
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
    if tensor.data.iter().any(|value| !value.is_finite()) {
        return Err(control_error(
            builtin,
            unsupported_model_identifier(builtin),
            format!("{builtin}: ss {name} must contain only finite real values"),
        ));
    }
    let mut matrix = DMatrix::<Complex64>::zeros(tensor.rows, tensor.cols);
    for col in 0..tensor.cols {
        for row in 0..tensor.rows {
            matrix[(row, col)] = Complex64::new(tensor.data[row + col * tensor.rows], 0.0);
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
            Ok(tensor
                .data
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect())
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(name, &tensor.shape, builtin)?;
            Ok(tensor
                .data
                .iter()
                .map(|(re, im)| Complex64::new(*re, *im))
                .collect())
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
        "isstable" => "RunMat:isstable:Internal",
        _ => "RunMat:tf:Internal",
    }
}

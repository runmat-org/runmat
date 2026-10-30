//! Shared SISO transfer-function object parsing, construction, and algebra.
mod admission;
#[cfg(test)]
mod boundary_tests;
mod domain;
mod numerics;
mod physical;
mod power;
pub use admission::preflight_ss_host_properties;
pub(super) use admission::state_matrix_dimensions;
use admission::{
    admit_coefficient_property, admit_delay_property, admit_sample_time_property,
    preflight_host_model_properties, preflight_ss_host_value,
};
pub use domain::{TfPolynomialBasis, TfSampleDomain, TfVariable};
use numerics::{finite_complex_ratio, finite_complex_ratio_direction};
pub use physical::PhysicalPolynomials;
use runmat_types::MemberAccess;

use std::collections::HashMap;

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_value::{CharArray, ComplexTensor, ObjectInstance, Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, dispatcher, BuiltinResult, RuntimeError};

pub const TF_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("tf");
pub const SS_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("ss");
pub const DEFAULT_CONTINUOUS_VARIABLE: &str = "s";
pub const DEFAULT_DISCRETE_VARIABLE: &str = "z";
pub const EPS: f64 = 1.0e-12;

#[derive(Clone, Copy)]
enum ArithmeticOperation {
    Plus,
    Minus,
    Multiply,
    Divide,
    Feedback,
}

impl ArithmeticOperation {
    fn name(self) -> &'static str {
        match self {
            Self::Plus => "plus",
            Self::Minus => "minus",
            Self::Multiply => "mtimes",
            Self::Divide => "mrdivide",
            Self::Feedback => "feedback",
        }
    }

    fn error(self, tf_identifier: &'static str) -> (&'static str, &'static str) {
        match self {
            Self::Feedback => ("feedback", "RunMat:feedback:UnsupportedModel"),
            _ => ("tf", tf_identifier),
        }
    }
}

static TF_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(TF_CLASS);

#[derive(Clone, Debug)]
pub struct TfModel {
    pub numerator: Vec<Complex64>,
    pub denominator: Vec<Complex64>,
    pub variable: TfVariable,
    pub sample_domain: TfSampleDomain,
    pub input_delay: f64,
    pub output_delay: f64,
}

#[derive(Clone, Debug)]
pub struct RealTfModel {
    pub numerator: Vec<f64>,
    pub denominator: Vec<f64>,
    pub sample_domain: TfSampleDomain,
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
    TF_CLASS_REGISTERED.ensure(|| {
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
                name.into(),
                crate::class_registry::RuntimeProperty {
                    name: name.into(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Public,
                    default_value: None,
                },
            );
        }

        let mut methods = HashMap::new();
        for method_name in [
            runmat_types::StaticMethodName::new("plus"),
            runmat_types::StaticMethodName::new("minus"),
            runmat_types::StaticMethodName::new("uplus"),
            runmat_types::StaticMethodName::new("uminus"),
            runmat_types::StaticMethodName::new("times"),
            runmat_types::StaticMethodName::new("mtimes"),
            runmat_types::StaticMethodName::new("rdivide"),
            runmat_types::StaticMethodName::new("mrdivide"),
            runmat_types::StaticMethodName::new("ldivide"),
            runmat_types::StaticMethodName::new("mldivide"),
            runmat_types::StaticMethodName::new("power"),
            runmat_types::StaticMethodName::new("mpower"),
        ] {
            methods.insert(
                method_name.into(),
                crate::class_registry::RuntimeMethod {
                    name: method_name.into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("{TF_CLASS}.{method_name}"),
                    implicit_class_argument: None,
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            );
        }

        crate::class_registry::register_builtin_class(
            TF_CLASS,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: TF_CLASS.into(),
                parent: None,
                properties,
                methods,
            },
        );
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
        let variable = TfVariable::parse(&options.variable, "tf")?;
        let sample_domain = TfSampleDomain::parse(options.sample_time, variable, "tf")?;
        validate_delay(input_delay, "InputDelay", "tf")?;
        validate_delay(output_delay, "OutputDelay", "tf")?;
        let (numerator, denominator) = physical_coefficients(numerator, denominator, variable);
        Ok(Self {
            numerator,
            denominator,
            variable,
            sample_domain,
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
        let variable = variable.into();
        let coefficients = if matches!(
            TfVariable::parse(&variable, "tf")?.basis(),
            TfPolynomialBasis::DiscreteInverse
        ) {
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
        } else {
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
        };
        Self::new(
            coefficients,
            vec![Complex64::new(1.0, 0.0)],
            TfOptions {
                variable,
                sample_time,
            },
        )
    }

    pub fn scalar(value: Complex64, options: TfOptions) -> BuiltinResult<Self> {
        Self::new(vec![value], vec![Complex64::new(1.0, 0.0)], options)
    }

    pub async fn from_value_async(value: Value, builtin: &'static str) -> BuiltinResult<Self> {
        let Value::Object(object) = value else {
            return Self::from_value(value, builtin);
        };
        if !object.is_class(TF_CLASS) {
            return Self::from_value(Value::Object(object), builtin);
        }

        // Check the complete required shape of the model before provider access.
        // Other object properties are not part of a transfer function's numeric
        // representation and must not trigger downloads or provider failures.
        const REQUIRED: [&str; 5] = [
            "Numerator",
            "Denominator",
            "Ts",
            "InputDelay",
            "OutputDelay",
        ];
        for name in REQUIRED.iter().copied() {
            property(&object, name, builtin)?;
        }
        preflight_host_model_properties(&object, builtin)?;
        let numerator =
            dispatcher::gather_if_needed_async(property(&object, "Numerator", builtin)?).await?;
        let denominator =
            dispatcher::gather_if_needed_async(property(&object, "Denominator", builtin)?).await?;
        let sample_time =
            dispatcher::gather_if_needed_async(property(&object, "Ts", builtin)?).await?;
        let input_delay =
            dispatcher::gather_if_needed_async(property(&object, "InputDelay", builtin)?).await?;
        let output_delay =
            dispatcher::gather_if_needed_async(property(&object, "OutputDelay", builtin)?).await?;
        let variable = match object.properties().get("Variable") {
            Some(value) => Some(dispatcher::gather_if_needed_async(value).await?),
            None => None,
        };
        Self::from_host_properties(
            &numerator,
            &denominator,
            variable.as_ref(),
            &sample_time,
            &input_delay,
            &output_delay,
            builtin,
        )
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
                    object.class_name()
                ),
            ));
        }

        Self::from_host_properties(
            property(&object, "Numerator", builtin)?,
            property(&object, "Denominator", builtin)?,
            object.properties().get("Variable"),
            property(&object, "Ts", builtin)?,
            property(&object, "InputDelay", builtin)?,
            property(&object, "OutputDelay", builtin)?,
            builtin,
        )
    }

    fn from_host_properties(
        numerator_value: &Value,
        denominator_value: &Value,
        variable_value: Option<&Value>,
        sample_time_value: &Value,
        input_delay_value: &Value,
        output_delay_value: &Value,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        admit_coefficient_property(numerator_value, "numerator", builtin)?;
        admit_coefficient_property(denominator_value, "denominator", builtin)?;
        let numerator = coefficients_from_value(numerator_value, "Numerator", builtin)?;
        let denominator = coefficients_from_value(denominator_value, "Denominator", builtin)?;
        validate_coefficients("numerator", &numerator, builtin)?;
        validate_coefficients("denominator", &denominator, builtin)?;
        admit_sample_time_property(sample_time_value, builtin)?;
        let sample_time = scalar_f64(sample_time_value, "Ts", builtin)?;
        admit_delay_property(input_delay_value, "InputDelay", builtin)?;
        admit_delay_property(output_delay_value, "OutputDelay", builtin)?;
        let input_delay = scalar_f64(input_delay_value, "InputDelay", builtin)?;
        let output_delay = scalar_f64(output_delay_value, "OutputDelay", builtin)?;
        validate_delay(input_delay, "InputDelay", builtin)?;
        validate_delay(output_delay, "OutputDelay", builtin)?;
        if all_zero(&denominator) {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: denominator coefficients must not all be zero"),
            ));
        }
        let variable = match variable_value {
            Some(value) => TfVariable::parse(&scalar_text(value, "Variable", builtin)?, builtin)?,
            None => {
                if sample_time != 0.0 {
                    TfVariable::Z
                } else {
                    TfVariable::S
                }
            }
        };
        let sample_domain = TfSampleDomain::parse(sample_time, variable, builtin)?;
        let (numerator, denominator) = physical_coefficients(numerator, denominator, variable);
        Ok(Self {
            numerator,
            denominator,
            variable,
            sample_domain,
            input_delay,
            output_delay,
        })
    }

    pub fn to_value(&self, builtin: &'static str) -> BuiltinResult<Value> {
        ensure_tf_class_registered();
        let (numerator, denominator) = self.external_coefficients();
        let mut object =
            runmat_value::UnboundedObjectBuilder::with_property_capacity(TF_CLASS.to_string(), 6);
        object.insert(
            "Numerator".to_string(),
            coefficient_value(&numerator, builtin)?,
        );
        object.insert(
            "Denominator".to_string(),
            coefficient_value(&denominator, builtin)?,
        );
        object.insert(
            "Variable".to_string(),
            Value::CharArray(CharArray::new_row(self.variable.as_str())),
        );
        object.insert("Ts".to_string(), Value::Num(self.sample_time()));
        object.insert("InputDelay".to_string(), Value::Num(self.input_delay));
        object.insert("OutputDelay".to_string(), Value::Num(self.output_delay));
        Ok(Value::Object(object.finish()))
    }

    pub fn coefficient_values(&self, builtin: &'static str) -> BuiltinResult<(Value, Value)> {
        let (numerator, denominator) = self.external_coefficients();
        Ok((
            coefficient_value(&numerator, builtin)?,
            coefficient_value(&denominator, builtin)?,
        ))
    }

    fn external_coefficients(&self) -> (Vec<Complex64>, Vec<Complex64>) {
        let mut numerator = self.numerator.clone();
        let mut denominator = self.denominator.clone();
        if self.variable.basis() == TfPolynomialBasis::DiscreteInverse {
            // Inverse-basis public arrays are ascending powers of z^-1.
            // Equal physical degree therefore requires leading, not trailing,
            // zero padding at egress. The two arrays must share one degree.
            let width = numerator.len().max(denominator.len());
            let numerator_pad = width - numerator.len();
            let denominator_pad = width - denominator.len();
            numerator.resize(width, Complex64::new(0.0, 0.0));
            denominator.resize(width, Complex64::new(0.0, 0.0));
            numerator.rotate_right(numerator_pad);
            denominator.rotate_right(denominator_pad);
        }
        (numerator, denominator)
    }

    pub fn to_real(&self, builtin: &'static str) -> BuiltinResult<RealTfModel> {
        let physical = self.physical_polynomials();
        let numerator = real_coefficients(physical.numerator(), "Numerator", builtin)?;
        let denominator = real_coefficients(physical.denominator(), "Denominator", builtin)?;
        Ok(RealTfModel {
            numerator,
            denominator,
            sample_domain: self.sample_domain,
            input_delay: self.input_delay,
            output_delay: self.output_delay,
        })
    }

    pub fn is_discrete(&self) -> bool {
        self.sample_domain.is_discrete()
    }

    pub fn sample_time(&self) -> f64 {
        self.sample_domain.sample_time()
    }

    pub fn sample_domain(&self) -> TfSampleDomain {
        self.sample_domain
    }

    pub fn ensure_zero_delays(&self, builtin: &'static str) -> BuiltinResult<()> {
        validate_zero_delays(self.input_delay, self.output_delay, builtin)
    }

    pub fn physical_polynomials(&self) -> PhysicalPolynomials<'_> {
        PhysicalPolynomials::borrowed(&self.numerator, &self.denominator)
    }

    pub fn normalized(&self) -> BuiltinResult<Self> {
        let leading = *self.denominator.first().ok_or_else(|| {
            control_error(
                "tf",
                "RunMat:tf:DenominatorInvalid",
                "tf: denominator coefficients cannot be empty",
            )
        })?;
        if leading.re == 0.0 && leading.im == 0.0 {
            return Err(control_error(
                "tf",
                "RunMat:tf:DenominatorInvalid",
                "tf: leading denominator coefficient must be non-zero",
            ));
        }
        let mut out = self.clone();
        out.numerator = out
            .numerator
            .iter()
            .map(|value| finite_complex_ratio(*value, leading))
            .collect();
        out.denominator = out
            .denominator
            .iter()
            .map(|value| finite_complex_ratio(*value, leading))
            .collect();
        if out
            .numerator
            .iter()
            .chain(&out.denominator)
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(control_error(
                "tf",
                "RunMat:tf:InvalidCoefficients",
                "tf: coefficient normalization exceeds the finite numeric range",
            ));
        }
        out.denominator[0] = Complex64::new(1.0, 0.0);
        Ok(out)
    }

    pub fn add(&self, rhs: &Self) -> BuiltinResult<Self> {
        self.ensure_arithmetic_compatible(rhs, ArithmeticOperation::Plus)?;
        let numerator = poly_add(
            &poly_mul(&self.numerator, &rhs.denominator),
            &poly_mul(&rhs.numerator, &self.denominator),
        );
        let denominator = poly_mul(&self.denominator, &rhs.denominator);
        self.with_new_coefficients(numerator, denominator)
    }

    pub fn sub(&self, rhs: &Self) -> BuiltinResult<Self> {
        self.ensure_arithmetic_compatible(rhs, ArithmeticOperation::Minus)?;
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
        self.ensure_arithmetic_compatible(rhs, ArithmeticOperation::Multiply)?;
        self.with_new_coefficients(
            poly_mul(&self.numerator, &rhs.numerator),
            poly_mul(&self.denominator, &rhs.denominator),
        )
    }

    pub fn div(&self, rhs: &Self) -> BuiltinResult<Self> {
        self.ensure_arithmetic_compatible(rhs, ArithmeticOperation::Divide)?;
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
        if exponent < 0 && all_zero(&self.numerator) {
            return Err(control_error(
                "tf",
                "RunMat:tf:DivideByZero",
                "tf: cannot divide by a zero transfer function",
            ));
        }
        power::preflight_power_capacity(self, exponent)?;
        let one = Self::scalar(
            Complex64::new(1.0, 0.0),
            TfOptions {
                variable: self.variable.as_str().to_string(),
                sample_time: self.sample_time(),
            },
        )?;
        if exponent == 0 {
            return Ok(one);
        }
        let mut base = self.clone();
        let mut exp = exponent.unsigned_abs();
        if exponent < 0 {
            base = one.div(&base)?;
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
        polynomial_roots(self.physical_polynomials().denominator(), "pole")
    }

    pub fn zeros(&self) -> BuiltinResult<Vec<Complex64>> {
        polynomial_roots(self.physical_polynomials().numerator(), "zero")
    }

    /// Evaluate an admitted model at a complex frequency-domain point.
    /// Exact denominator zeros are undefined here; DC pole direction is
    /// handled separately by `dc_gain`.
    pub fn frequency_response(&self, point: Complex64) -> Complex64 {
        self.physical_polynomials()
            .evaluate(point)
            .unwrap_or(Complex64::new(f64::NAN, f64::NAN))
    }

    pub fn dc_gain(&self) -> BuiltinResult<Complex64> {
        let point = if self.is_discrete() {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
        let physical = self.physical_polynomials();
        let num = poly_eval(physical.numerator(), point);
        let den = poly_eval(physical.denominator(), point);
        if den == Complex64::new(0.0, 0.0) {
            if num == Complex64::new(0.0, 0.0) {
                Ok(Complex64::new(f64::NAN, f64::NAN))
            } else {
                // Direct division by a zero complex denominator produces a spurious NaN
                // component. Factor the pole so the infinite result retains the local
                // signed/complex-axis direction of the positive-real approach to the DC point.
                let local_denominator =
                    first_nonzero_local_polynomial_coefficient(physical.denominator(), point);
                Ok(complex_infinity_in_direction(
                    finite_complex_ratio_direction(num, local_denominator),
                ))
            }
        } else {
            Ok(finite_complex_ratio(num, den))
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
        self.ensure_arithmetic_compatible(rhs, ArithmeticOperation::Feedback)?;
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

    fn ensure_arithmetic_compatible(
        &self,
        rhs: &Self,
        op: ArithmeticOperation,
    ) -> BuiltinResult<()> {
        let name = op.name();
        if self.sample_domain != rhs.sample_domain {
            let (builtin, identifier) = op.error("RunMat:tf:SampleTimeMismatch");
            return Err(control_error(
                builtin,
                identifier,
                format!("{builtin}.{name}: sample times must match"),
            ));
        }
        if self.variable != rhs.variable {
            let (builtin, identifier) = op.error("RunMat:tf:VariableMismatch");
            return Err(control_error(
                builtin,
                identifier,
                format!("{builtin}.{name}: transfer-function variables must match"),
            ));
        }
        if self.input_delay != 0.0
            || self.output_delay != 0.0
            || rhs.input_delay != 0.0
            || rhs.output_delay != 0.0
        {
            let (builtin, identifier) = op.error("RunMat:tf:UnsupportedDelay");
            return Err(control_error(
                builtin,
                identifier,
                format!(
                    "{builtin}.{name}: input and output delays are not supported in arithmetic"
                ),
            ));
        }
        Ok(())
    }

    fn with_new_coefficients(
        &self,
        numerator: Vec<Complex64>,
        denominator: Vec<Complex64>,
    ) -> BuiltinResult<Self> {
        validate_coefficients("numerator", &numerator, "tf")?;
        validate_coefficients("denominator", &denominator, "tf")?;
        if all_zero(&denominator) {
            return Err(control_error(
                "tf",
                "RunMat:tf:DenominatorInvalid",
                "tf: denominator coefficients must not all be zero",
            ));
        }
        Ok(Self {
            numerator: trim_leading_complex_zeros(numerator),
            denominator: trim_leading_complex_zeros(denominator),
            variable: self.variable,
            sample_domain: self.sample_domain,
            input_delay: self.input_delay,
            output_delay: self.output_delay,
        })
    }
}

impl RealTfModel {
    pub fn sample_time(&self) -> f64 {
        self.sample_domain.sample_time()
    }

    pub fn normalized(&self, builtin: &'static str) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
        let den = trim_leading_real_zeros(self.denominator.clone());
        if den.is_empty() || den[0] == 0.0 {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: leading denominator coefficient must be non-zero"),
            ));
        }
        let num = trim_leading_real_zeros(self.numerator.clone());
        let leading = den[0];
        let normalized_num: Vec<f64> = num.iter().map(|value| value / leading).collect();
        let mut normalized_den: Vec<f64> = den.iter().map(|value| value / leading).collect();
        if normalized_num
            .iter()
            .chain(&normalized_den)
            .any(|value| !value.is_finite())
        {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: coefficient normalization exceeds the finite numeric range"),
            ));
        }
        normalized_den[0] = 1.0;
        Ok((normalized_num, normalized_den))
    }

    pub fn ensure_zero_delays(&self, builtin: &'static str) -> BuiltinResult<()> {
        validate_zero_delays(self.input_delay, self.output_delay, builtin)
    }
}

fn validate_zero_delays(input: f64, output: f64, builtin: &'static str) -> BuiltinResult<()> {
    if input != 0.0 || output != 0.0 {
        return Err(control_error(
            builtin,
            unsupported_model_identifier(builtin),
            format!("{builtin}: transfer functions with input or output delays are not supported"),
        ));
    }
    Ok(())
}

pub async fn parse_coefficients(
    label: &str,
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<Vec<Complex64>> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    let coeffs = match gathered {
        Value::Tensor(tensor) => {
            ensure_vector_shape(label, tensor.shape(), builtin)?;
            tensor::tensor_values_f64(&tensor)
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect()
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(label, tensor.shape(), builtin)?;
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
            ensure_vector_shape(label, tensor.shape(), builtin)?;
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
    if matches!(value, Value::Object(_)) {
        return TfModel::from_value_async(value, builtin).await;
    }
    let gathered = gather_direct_resident(value).await?;
    let scalar = scalar_complex(&gathered, builtin)?;
    TfModel::scalar(
        scalar,
        TfOptions {
            variable: reference.variable.as_str().to_string(),
            sample_time: reference.sample_time(),
        },
    )
}

async fn gather_direct_resident(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        dispatcher::gather_if_needed_async(&value).await
    } else {
        Ok(value)
    }
}

pub async fn two_models_ordered(
    lhs: Value,
    rhs: Value,
    builtin: &'static str,
) -> BuiltinResult<(TfModel, TfModel)> {
    preflight_model_operand(&lhs, builtin)?;
    preflight_model_operand(&rhs, builtin)?;
    // Parse object operands before preparing the other operand. Gathering the
    // pair here would recursively download unrelated model properties, and
    // could touch the second provider before the first model is validated.
    match (lhs, rhs) {
        (left @ Value::Object(_), right @ Value::Object(_)) => Ok((
            TfModel::from_value_async(left, builtin).await?,
            TfModel::from_value_async(right, builtin).await?,
        )),
        (left @ Value::Object(_), right) => {
            let left_model = TfModel::from_value_async(left, builtin).await?;
            let right_model = value_to_model_with_reference(right, &left_model, builtin).await?;
            Ok((left_model, right_model))
        }
        (left, right @ Value::Object(_)) => {
            let right_model = TfModel::from_value_async(right, builtin).await?;
            let left_model = value_to_model_with_reference(left, &right_model, builtin).await?;
            Ok((left_model, right_model))
        }
        (left, right) => {
            let left = gather_direct_resident(left).await?;
            let right = gather_direct_resident(right).await?;
            let options = TfOptions::default();
            Ok((
                TfModel::scalar(scalar_complex(&left, builtin)?, options.clone())?,
                TfModel::scalar(scalar_complex(&right, builtin)?, options)?,
            ))
        }
    }
}

pub fn preflight_model_operand(value: &Value, builtin: &'static str) -> BuiltinResult<()> {
    match value {
        Value::Object(object) => {
            if !object.is_class(TF_CLASS) {
                return Err(control_error(
                    builtin,
                    unsupported_model_identifier(builtin),
                    format!(
                        "{builtin}: unsupported model class '{}'",
                        object.class_name()
                    ),
                ));
            }
            for name in [
                "Numerator",
                "Denominator",
                "Ts",
                "InputDelay",
                "OutputDelay",
            ] {
                property(object, name, builtin)?;
            }
            preflight_host_model_properties(object, builtin)
        }
        Value::GpuTensor(_) => Ok(()),
        value => {
            let scalar = scalar_complex(value, builtin)?;
            validate_coefficients("scalar gain", &[scalar], builtin)
        }
    }
}

pub fn scalar_text(
    value: &Value,
    context: &str,
    builtin: &'static str,
) -> BuiltinResult<runmat_value::Utf8Text> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.clone()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("nonmissing scalar text")
                .clone())
        }
        Value::CharArray(array) if array.rows == 1 => {
            Ok(runmat_value::decode_utf16_code_units(array.data())
                .map_err(|source| {
                    let mut error = control_error(
                        builtin,
                        invalid_argument_identifier(builtin),
                        source.to_string(),
                    );
                    error.source = Some(Box::new(source));
                    error
                })?
                .into())
        }
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
    Ok(TfVariable::parse(variable, builtin)?.as_str().to_string())
}

pub fn is_discrete_variable(variable: &str) -> bool {
    matches!(TfVariable::parse(variable, "tf"), Ok(value) if value.is_discrete())
}

pub fn validate_variable_domain(
    variable: &str,
    sample_time: f64,
    builtin: &'static str,
) -> BuiltinResult<()> {
    TfSampleDomain::parse(sample_time, TfVariable::parse(variable, builtin)?, builtin).map(|_| ())
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
    // Scalar gains are data, not a numerical pole classification: a small but
    // nonzero imaginary component must survive the output boundary exactly.
    if value.im == 0.0 {
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
    preflight_ss_host_properties(object, &["A", "Ts"], builtin)?;
    ss_poles_from_values(
        property(object, "A", builtin)?,
        property(object, "Ts", builtin)?,
        builtin,
    )
}

fn ss_poles_from_values(
    state: &Value,
    sample: &Value,
    builtin: &'static str,
) -> BuiltinResult<(Vec<Complex64>, f64)> {
    let a = ss_state_matrix_value(state, "A", builtin)?;
    let sample_time = scalar_f64(sample, "Ts", builtin)?;
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

/// Gather only the state-space properties consumed by an analysis operation.
/// The fixed result lives in the caller's future; no temporary Object
/// generation or property map is allocated merely to parse host values.
pub async fn gather_ss_property_values<const N: usize>(
    object: &ObjectInstance,
    required: [&'static str; N],
    builtin: &'static str,
) -> BuiltinResult<[Value; N]> {
    for name in required {
        if !object.properties().contains_key(name) {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: ss object is missing {name}"),
            ));
        }
    }
    preflight_ss_host_properties(object, &required, builtin)?;
    let mut values = required.map(|name| {
        object
            .properties()
            .get(name)
            .expect("preflight checked property")
            .clone()
    });
    let mut attempted = false;
    for (name, value) in required.into_iter().zip(&mut values) {
        attempted |= matches!(value, Value::GpuTensor(_));
        let terminal = |mut error: RuntimeError| {
            if attempted {
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            }
            error
        };
        *value = dispatcher::gather_if_needed_async(value)
            .await
            .map_err(terminal)?;
        preflight_ss_host_value(name, value, builtin).map_err(terminal)?;
    }
    Ok(values)
}

/// Gather only the state matrix and sample time used by pole analysis. Other
/// state-space properties may contain resident values but are not read here.
pub async fn ss_poles_from_object_async(
    object: &ObjectInstance,
    builtin: &'static str,
) -> BuiltinResult<(Vec<Complex64>, f64)> {
    for name in ["A", "Ts"] {
        if !object.properties().contains_key(name) {
            return Err(control_error(
                builtin,
                invalid_model_identifier(builtin),
                format!("{builtin}: ss object is missing {name}"),
            ));
        }
    }
    preflight_ss_host_properties(object, &["A", "Ts"], builtin)?;
    let a_resident = matches!(object.properties().get("A"), Some(Value::GpuTensor(_)));
    let ts_resident = matches!(object.properties().get("Ts"), Some(Value::GpuTensor(_)));
    if !a_resident {
        // Pole analysis accepts only real A, unlike general SS conversion.
        ss_state_matrix_property(object, "A", builtin)?;
    }
    if !a_resident && !ts_resident {
        return ss_poles_from_object(object, builtin);
    }
    // Reuse the consumed-property owner: native integer payload admission is
    // repeated after its one owning gather, before any binary64 conversion.
    let [state, sample] = gather_ss_property_values(object, ["A", "Ts"], builtin).await?;
    ss_poles_from_values(&state, &sample, builtin).map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

fn ss_state_matrix_property(
    object: &ObjectInstance,
    name: &'static str,
    builtin: &'static str,
) -> BuiltinResult<DMatrix<Complex64>> {
    let value = object.properties().get(name).ok_or_else(|| {
        control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: ss object is missing {name}"),
        )
    })?;
    ss_state_matrix_value(value, name, builtin)
}

fn ss_state_matrix_value(
    value: &Value,
    name: &'static str,
    builtin: &'static str,
) -> BuiltinResult<DMatrix<Complex64>> {
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
    let (rows, cols) = state_matrix_dimensions(&tensor);
    if tensor.shape().len() > 2 || rows != cols {
        return Err(control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!(
                "{builtin}: ss {name} must be square, got {:?}",
                tensor.shape()
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
    let mut matrix = DMatrix::<Complex64>::zeros(rows, cols);
    for col in 0..cols {
        for row in 0..rows {
            matrix[(row, col)] = Complex64::new(values[row + col * rows], 0.0);
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
    if leading.re == 0.0 && leading.im == 0.0 {
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
    while quotient.len() > 1 && poly_eval(&quotient, point) == Complex64::new(0.0, 0.0) {
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
        if value == 0.0 {
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
        .position(|value| *value != 0.0)
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
    let value = object.properties().get(name).ok_or_else(|| {
        control_error(
            builtin,
            invalid_model_identifier(builtin),
            format!("{builtin}: tf object is missing {name}"),
        )
    })?;
    coefficients_from_value(value, name, builtin)
}

fn coefficients_from_value(
    value: &Value,
    name: &str,
    builtin: &'static str,
) -> BuiltinResult<Vec<Complex64>> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(name, tensor.shape(), builtin)?;
            Ok(tensor::tensor_values_f64(tensor)
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect())
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(name, tensor.shape(), builtin)?;
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

pub(crate) fn coefficient_value(
    coeffs: &[Complex64],
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let len = coeffs.len();
    if coeffs.iter().all(|coeff| coeff.im == 0.0) {
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
    object.properties().get(name).ok_or_else(|| {
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
        if coeff.im != 0.0 {
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

fn physical_coefficients(
    mut numerator: Vec<Complex64>,
    mut denominator: Vec<Complex64>,
    variable: TfVariable,
) -> (Vec<Complex64>, Vec<Complex64>) {
    if variable.basis() == TfPolynomialBasis::DiscreteInverse {
        // Public inverse-basis arrays are ascending in z^-1. With a *shared*
        // degree they are already descending physical-z coefficients: multiply
        // both polynomials by the same z power by padding their right edges.
        let width = numerator.len().max(denominator.len());
        numerator.resize(width, Complex64::new(0.0, 0.0));
        denominator.resize(width, Complex64::new(0.0, 0.0));
    }
    (
        trim_leading_complex_zeros(numerator),
        trim_leading_complex_zeros(denominator),
    )
}

fn trim_leading_complex_zeros(mut coeffs: Vec<Complex64>) -> Vec<Complex64> {
    let first_nonzero = coeffs
        .iter()
        .position(|value| value.re != 0.0 || value.im != 0.0)
        .unwrap_or(coeffs.len());
    coeffs.drain(..first_nonzero);
    if coeffs.is_empty() {
        vec![Complex64::new(0.0, 0.0)]
    } else {
        coeffs
    }
}

fn all_zero(coeffs: &[Complex64]) -> bool {
    coeffs
        .iter()
        .all(|coeff| coeff.re == 0.0 && coeff.im == 0.0)
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
        "nyquist" => "RunMat:nyquist:InvalidArgument",
        "step" => "RunMat:step:InvalidArgument",
        "impulse" => "RunMat:impulse:InvalidArgument",
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
        "nyquist" => "RunMat:nyquist:InvalidModel",
        "step" => "RunMat:step:InvalidModel",
        "impulse" => "RunMat:impulse:InvalidModel",
        "feedback" => "RunMat:feedback:InvalidModel",
        "stepinfo" => "RunMat:stepinfo:InvalidData",
        "dcgain" => "RunMat:dcgain:InvalidModel",
        "pole" => "RunMat:pole:InvalidModel",
        "zero" => "RunMat:zero:InvalidModel",
        "damp" => "RunMat:damp:InvalidModel",
        "rlocus" => "RunMat:rlocus:InvalidModel",
        "pzmap" => "RunMat:pzmap:InvalidModel",
        "isstable" => "RunMat:isstable:InvalidModel",
        "tfdata" => "RunMat:tfdata:InvalidModel",
        "series" => "RunMat:series:InvalidModel",
        "ss2tf" => "RunMat:ss2tf:InvalidModel",
        _ => "RunMat:tf:InvalidCoefficients",
    }
}

fn invalid_sample_time_identifier(builtin: &str) -> &'static str {
    match builtin {
        "nyquist" => "RunMat:nyquist:InvalidModel",
        "step" => "RunMat:step:InvalidModel",
        "impulse" => "RunMat:impulse:InvalidModel",
        "feedback" => "RunMat:feedback:InvalidSampleTime",
        "lqr" => "RunMat:lqr:InvalidModel",
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

fn invalid_variable_identifier(builtin: &str) -> &'static str {
    match builtin {
        "nyquist" => "RunMat:nyquist:InvalidModel",
        "stepinfo" => "RunMat:stepinfo:InvalidSystem",
        "step" => "RunMat:step:InvalidModel",
        "impulse" => "RunMat:impulse:InvalidModel",
        "feedback" => "RunMat:feedback:InvalidVariable",
        _ => "RunMat:tf:InvalidVariable",
    }
}

pub(crate) fn invalid_model_identifier(builtin: &str) -> &'static str {
    match builtin {
        "nyquist" => "RunMat:nyquist:InvalidModel",
        "step" => "RunMat:step:InvalidModel",
        "impulse" => "RunMat:impulse:InvalidModel",
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

pub(crate) fn unsupported_model_identifier(builtin: &str) -> &'static str {
    match builtin {
        "nyquist" => "RunMat:nyquist:UnsupportedModel",
        "step" => "RunMat:step:UnsupportedModel",
        "impulse" => "RunMat:impulse:UnsupportedModel",
        "feedback" => "RunMat:feedback:UnsupportedModel",
        "stepinfo" => "RunMat:stepinfo:UnsupportedModel",
        "dcgain" => "RunMat:dcgain:UnsupportedModel",
        "pole" => "RunMat:pole:UnsupportedModel",
        "zero" => "RunMat:zero:UnsupportedModel",
        "damp" => "RunMat:damp:UnsupportedModel",
        "rlocus" => "RunMat:rlocus:UnsupportedModel",
        "pzmap" => "RunMat:pzmap:UnsupportedModel",
        "isstable" => "RunMat:isstable:UnsupportedModel",
        "tfdata" => "RunMat:tfdata:UnsupportedModel",
        "series" => "RunMat:series:UnsupportedModel",
        "ss2tf" => "RunMat:ss2tf:UnsupportedModel",
        _ => "RunMat:tf:UnsupportedModel",
    }
}

pub(crate) fn internal_identifier(builtin: &str) -> &'static str {
    match builtin {
        "nyquist" => "RunMat:nyquist:Internal",
        "step" => "RunMat:step:Internal",
        "impulse" => "RunMat:impulse:Internal",
        "feedback" => "RunMat:feedback:Internal",
        "stepinfo" => "RunMat:stepinfo:Internal",
        "dcgain" => "RunMat:dcgain:Internal",
        "pole" => "RunMat:pole:Internal",
        "zero" => "RunMat:zero:Internal",
        "damp" => "RunMat:damp:Internal",
        "rlocus" => "RunMat:rlocus:Internal",
        "pzmap" => "RunMat:pzmap:Internal",
        "isstable" => "RunMat:isstable:Internal",
        "tfdata" => "RunMat:tfdata:Internal",
        "series" => "RunMat:series:Internal",
        "ss2tf" => "RunMat:ss2tf:Internal",
        _ => "RunMat:tf:Internal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntegerComplexStorage, IntegerStorage};

    fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn tiny_nonzero_coefficients_and_imaginary_parts_survive_model_roundtrip() {
        let tiny = 1.0e-14;
        let model = TfModel::new(
            vec![Complex64::new(0.0, tiny), Complex64::new(tiny, 0.0)],
            vec![Complex64::new(tiny, 0.0), Complex64::new(1.0, 0.0)],
            TfOptions::default(),
        )
        .expect("tiny coefficients are nonzero");
        let Value::Object(object) = model.to_value("feedback").expect("serialize") else {
            panic!("model object")
        };
        let Value::ComplexTensor(numerator) = object.properties().get("Numerator").unwrap() else {
            panic!("complex numerator")
        };
        assert_eq!(numerator.shape(), [1, 2]);
        assert_eq!(
            tensor::complex_tensor_value_complex64(numerator, 0).im,
            tiny
        );
        let restored = TfModel::from_value(Value::Object(object), "feedback").expect("roundtrip");
        assert_eq!(restored.numerator[0].im, tiny);
        assert_eq!(restored.denominator[0].re, tiny);
    }

    #[test]
    fn scaled_polynomial_roots_do_not_treat_tiny_leading_coefficient_as_zero() {
        let scale = 1.0e-14;
        let roots = polynomial_roots(
            &[
                Complex64::new(scale, 0.0),
                Complex64::new(3.0 * scale, 0.0),
                Complex64::new(2.0 * scale, 0.0),
            ],
            "pole",
        )
        .expect("nonzero leading coefficient");
        assert_eq!(roots.len(), 2);
        assert!(roots
            .iter()
            .any(|root| (*root - Complex64::new(-1.0, 0.0)).norm() < 1.0e-9));
        assert!(roots
            .iter()
            .any(|root| (*root - Complex64::new(-2.0, 0.0)).norm() < 1.0e-9));
    }

    #[test]
    fn feedback_reports_its_own_structural_compatibility_errors() {
        let lhs = TfModel::scalar(Complex64::new(1.0, 0.0), TfOptions::default()).unwrap();
        let mut rhs = TfModel::scalar(
            Complex64::new(1.0, 0.0),
            TfOptions {
                variable: "z".into(),
                sample_time: 1.0e-14,
            },
        )
        .unwrap();
        assert_eq!(
            lhs.feedback(&rhs, -1.0).unwrap_err().identifier(),
            Some("RunMat:feedback:UnsupportedModel")
        );
        rhs = TfModel::scalar(
            Complex64::new(1.0, 0.0),
            TfOptions {
                variable: "p".into(),
                sample_time: 0.0,
            },
        )
        .unwrap();
        assert_eq!(
            lhs.feedback(&rhs, -1.0).unwrap_err().identifier(),
            Some("RunMat:feedback:UnsupportedModel")
        );
        rhs = lhs.clone();
        rhs.input_delay = 1.0e-14;
        assert_eq!(
            lhs.feedback(&rhs, -1.0).unwrap_err().identifier(),
            Some("RunMat:feedback:UnsupportedModel")
        );
    }

    #[test]
    fn ss_host_sample_time_is_rejected_before_resident_state_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &a).unwrap();
            let mut object = runmat_value::UnboundedObjectBuilder::new(SS_CLASS);
            object.insert("A".into(), Value::GpuTensor(resident));
            object.insert("Ts".into(), Value::String("bad".into()));
            let object = object.finish();
            provider.reset_telemetry();
            let error =
                block_on(ss_poles_from_object_async(&object, "pole")).expect_err("invalid Ts");
            assert_eq!(error.identifier(), Some("RunMat:pole:InvalidArgument"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    #[test]
    fn pole_real_state_rule_precedes_resident_sample_time_download() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let ts = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let resident =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &ts).unwrap();
            let mut object = runmat_value::UnboundedObjectBuilder::new(SS_CLASS);
            object.insert("A".into(), Value::Complex(1.0, 1.0));
            object.insert("Ts".into(), Value::GpuTensor(resident));
            let object = object.finish();
            provider.reset_telemetry();
            let error = block_on(ss_poles_from_object_async(&object, "pole"))
                .expect_err("complex A not supported by pole parser");
            assert_eq!(error.identifier(), Some("RunMat:pole:UnsupportedModel"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        });
    }

    fn poisoned_complex_integer_tensor(
        real: IntegerStorage,
        imag: IntegerStorage,
        shape: Vec<usize>,
    ) -> Value {
        let storage = IntegerComplexStorage::new(real, imag).expect("complex integer storage");
        let tensor = ComplexTensor::new_integer(storage, shape).expect("complex integer tensor");
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
        let mut object = runmat_value::UnboundedObjectBuilder::new(TF_CLASS);
        object.insert(
            "Numerator".to_string(),
            poisoned_complex_integer_tensor(
                IntegerStorage::I16(vec![5, 7]),
                IntegerStorage::I16(vec![-6, 8]),
                vec![1, 2],
            ),
        );
        let object = object.finish();

        assert_eq!(
            coefficients_from_property(&object, "Numerator", "tf").expect("coefficients"),
            vec![Complex64::new(5.0, -6.0), Complex64::new(7.0, 8.0)]
        );
    }

    #[test]
    fn ss_state_matrix_property_reads_typed_integer_storage_exactly() {
        let mut object = runmat_value::UnboundedObjectBuilder::new(SS_CLASS);
        object.insert(
            "A".to_string(),
            poisoned_integer_tensor(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2]),
        );
        let object = object.finish();

        let matrix = ss_state_matrix_property(&object, "A", "pole").expect("state matrix");
        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix[(0, 0)], Complex64::new(1.0, 0.0));
        assert_eq!(matrix[(1, 0)], Complex64::new(3.0, 0.0));
        assert_eq!(matrix[(0, 1)], Complex64::new(2.0, 0.0));
        assert_eq!(matrix[(1, 1)], Complex64::new(4.0, 0.0));
    }
}

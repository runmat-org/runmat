//! MATLAB-compatible `tf` transfer-function constructor for RunMat.

use std::collections::HashMap;
use std::sync::OnceLock;

use num_complex::Complex64;
use runmat_builtins::{
    Access, CharArray, ClassDef, ComplexTensor, MethodDef, ObjectInstance, PropertyDef, Tensor,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::control::type_resolvers::tf_type;
use crate::{build_runtime_error, dispatcher, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "tf";
const TF_CLASS: &str = "tf";
const DEFAULT_VARIABLE: &str = "s";
const EPS: f64 = 1.0e-12;

static TF_CLASS_REGISTERED: OnceLock<()> = OnceLock::new();

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::tf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tf",
    op_kind: GpuOpKind::Custom("transfer-function-constructor"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Object construction runs on the host. gpuArray coefficient inputs are gathered before storing the transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::tf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Transfer-function construction is metadata-only and terminates numeric fusion chains.",
};

fn tf_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn ensure_tf_class_registered() {
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
                    is_dependent: false,
                    get_access: Access::Public,
                    set_access: Access::Public,
                    default_value: None,
                },
            );
        }

        let methods: HashMap<String, MethodDef> = HashMap::new();
        runmat_builtins::register_class(ClassDef {
            name: TF_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
    });
}

#[runtime_builtin(
    name = "tf",
    category = "control",
    summary = "Create a SISO transfer-function object from numerator and denominator coefficient vectors.",
    keywords = "tf,transfer function,control system,filter,polynomial",
    type_resolver(tf_type),
    builtin_path = "crate::builtins::control::tf"
)]
async fn tf_builtin(
    numerator: Value,
    denominator: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let options = TfOptions::parse(&rest)?;
    let numerator = Coefficients::parse("numerator", numerator).await?;
    let denominator = Coefficients::parse("denominator", denominator).await?;

    if denominator.coeffs.is_empty() {
        return Err(tf_error("tf: denominator coefficients cannot be empty"));
    }
    if denominator.is_all_zero() {
        return Err(tf_error(
            "tf: denominator coefficients must not all be zero",
        ));
    }

    ensure_tf_class_registered();
    let mut object = ObjectInstance::new(TF_CLASS.to_string());
    object
        .properties
        .insert("Numerator".to_string(), numerator.into_row_value()?);
    object
        .properties
        .insert("Denominator".to_string(), denominator.into_row_value()?);
    object.properties.insert(
        "Variable".to_string(),
        Value::CharArray(CharArray::new_row(&options.variable)),
    );
    object
        .properties
        .insert("Ts".to_string(), Value::Num(options.sample_time));
    object
        .properties
        .insert("InputDelay".to_string(), Value::Num(0.0));
    object
        .properties
        .insert("OutputDelay".to_string(), Value::Num(0.0));
    Ok(Value::Object(object))
}

#[derive(Clone)]
struct TfOptions {
    variable: String,
    sample_time: f64,
}

impl TfOptions {
    fn parse(rest: &[Value]) -> BuiltinResult<Self> {
        let mut options = Self {
            variable: DEFAULT_VARIABLE.to_string(),
            sample_time: 0.0,
        };

        match rest {
            [] => {}
            [sample_time] => {
                options.sample_time = parse_sample_time(sample_time)?;
                options.variable = "z".to_string();
            }
            _ => {
                if !rest.len().is_multiple_of(2) {
                    return Err(tf_error(
                        "tf: optional arguments must be name-value pairs or a scalar sample time",
                    ));
                }
                let mut idx = 0;
                while idx < rest.len() {
                    let name = scalar_text(&rest[idx], "option name")?;
                    let lowered = name.trim().to_ascii_lowercase();
                    let value = &rest[idx + 1];
                    match lowered.as_str() {
                        "variable" => options.variable = parse_variable(value)?,
                        "ts" | "sampletime" => options.sample_time = parse_sample_time(value)?,
                        _ => {
                            return Err(tf_error(format!("tf: unsupported option '{name}'")));
                        }
                    }
                    idx += 2;
                }
                if options.sample_time > 0.0 && options.variable == DEFAULT_VARIABLE {
                    options.variable = "z".to_string();
                }
            }
        }

        Ok(options)
    }
}

fn parse_sample_time(value: &Value) -> BuiltinResult<f64> {
    let sample_time = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        other => {
            return Err(tf_error(format!(
                "tf: sample time must be a non-negative scalar, got {other:?}"
            )))
        }
    };
    if !sample_time.is_finite() || sample_time < 0.0 {
        return Err(tf_error(
            "tf: sample time must be a finite non-negative scalar",
        ));
    }
    Ok(sample_time)
}

fn parse_variable(value: &Value) -> BuiltinResult<String> {
    let variable = scalar_text(value, "Variable")?;
    let variable = variable.trim();
    match variable {
        "s" | "p" | "z" | "q" | "z^-1" | "q^-1" => Ok(variable.to_string()),
        _ => Err(tf_error(
            "tf: Variable must be one of 's', 'p', 'z', 'q', 'z^-1', or 'q^-1'",
        )),
    }
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        other => Err(tf_error(format!(
            "tf: {context} must be a string scalar or character vector, got {other:?}"
        ))),
    }
}

#[derive(Clone)]
struct Coefficients {
    coeffs: Vec<Complex64>,
}

impl Coefficients {
    async fn parse(label: &str, value: Value) -> BuiltinResult<Self> {
        let gathered = dispatcher::gather_if_needed_async(&value).await?;
        let coeffs = match gathered {
            Value::Tensor(tensor) => {
                ensure_vector_shape(label, &tensor.shape)?;
                tensor
                    .data
                    .into_iter()
                    .map(|re| Complex64::new(re, 0.0))
                    .collect()
            }
            Value::ComplexTensor(tensor) => {
                ensure_vector_shape(label, &tensor.shape)?;
                tensor
                    .data
                    .into_iter()
                    .map(|(re, im)| Complex64::new(re, im))
                    .collect()
            }
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical).map_err(tf_error)?;
                ensure_vector_shape(label, &tensor.shape)?;
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
                return Err(tf_error(format!(
                    "tf: {label} must be a numeric coefficient vector, got {other:?}"
                )));
            }
        };

        if coeffs.is_empty() {
            return Err(tf_error(format!(
                "tf: {label} coefficients cannot be empty"
            )));
        }
        for coeff in &coeffs {
            if !coeff.re.is_finite() || !coeff.im.is_finite() {
                return Err(tf_error(format!("tf: {label} coefficients must be finite")));
            }
        }

        Ok(Self { coeffs })
    }

    fn is_all_zero(&self) -> bool {
        self.coeffs.iter().all(|coeff| coeff.norm() <= EPS)
    }

    fn into_row_value(self) -> BuiltinResult<Value> {
        let len = self.coeffs.len();
        if self.coeffs.iter().all(|coeff| coeff.im.abs() <= EPS) {
            let data = self.coeffs.into_iter().map(|coeff| coeff.re).collect();
            let tensor =
                Tensor::new(data, vec![1, len]).map_err(|err| tf_error(format!("tf: {err}")))?;
            Ok(Value::Tensor(tensor))
        } else {
            let data = self
                .coeffs
                .into_iter()
                .map(|coeff| (coeff.re, coeff.im))
                .collect();
            let tensor = ComplexTensor::new(data, vec![1, len])
                .map_err(|err| tf_error(format!("tf: {err}")))?;
            Ok(Value::ComplexTensor(tensor))
        }
    }
}

fn ensure_vector_shape(label: &str, shape: &[usize]) -> BuiltinResult<()> {
    let non_unit = shape.iter().copied().filter(|&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(tf_error(format!(
            "tf: {label} coefficients must be a vector"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntValue;

    fn run_tf(numerator: Value, denominator: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(tf_builtin(numerator, denominator, rest))
    }

    fn property<'a>(value: &'a Value, name: &str) -> &'a Value {
        let Value::Object(object) = value else {
            panic!("expected object, got {value:?}");
        };
        object
            .properties
            .get(name)
            .unwrap_or_else(|| panic!("missing property {name}"))
    }

    #[test]
    fn tf_constructs_continuous_siso_object() {
        let sys = run_tf(
            Value::Num(20.0),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect("tf");

        let Value::Object(object) = &sys else {
            panic!("expected object");
        };
        assert_eq!(object.class_name, "tf");
        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("s"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.0));
        match property(&sys, "Numerator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.data, vec![20.0]);
            }
            other => panic!("expected numerator tensor, got {other:?}"),
        }
        match property(&sys, "Denominator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![1.0, 5.0]);
            }
            other => panic!("expected denominator tensor, got {other:?}"),
        }
    }

    #[test]
    fn tf_normalizes_column_coefficients_to_rows() {
        let sys = run_tf(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0], vec![3, 1]).unwrap()),
            Vec::new(),
        )
        .expect("tf");

        match property(&sys, "Numerator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![1.0, 2.0]);
            }
            other => panic!("expected numerator tensor, got {other:?}"),
        }
        match property(&sys, "Denominator") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.data, vec![1.0, 3.0, 2.0]);
            }
            other => panic!("expected denominator tensor, got {other:?}"),
        }
    }

    #[test]
    fn tf_accepts_discrete_sample_time() {
        let sys = run_tf(
            Value::Int(IntValue::I32(1)),
            Value::Tensor(Tensor::new(vec![1.0, -0.5], vec![1, 2]).unwrap()),
            vec![Value::Num(0.1)],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("z"))
        );
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.1));
    }

    #[test]
    fn tf_accepts_variable_name_value_option() {
        let sys = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            vec![Value::from("Variable"), Value::from("p")],
        )
        .expect("tf");

        assert_eq!(
            property(&sys, "Variable"),
            &Value::CharArray(CharArray::new_row("p"))
        );
    }

    #[test]
    fn tf_rejects_zero_denominator() {
        let err = run_tf(
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect_err("zero denominator should fail");
        assert!(err.message().contains("must not all be zero"));
    }

    #[test]
    fn tf_rejects_matrix_coefficients() {
        let err = run_tf(
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 5.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect_err("matrix numerator should fail");
        assert!(err
            .message()
            .contains("numerator coefficients must be a vector"));
    }
}

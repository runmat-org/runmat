//! MATLAB-compatible `ss` state-space model constructor for RunMat.

use std::cell::Cell;
use std::collections::HashMap;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ClassDef, MethodDef, ObjectInstance, PropertyDef, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::type_resolvers::ss_type;
use crate::{build_runtime_error, dispatcher, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "ss";
const SS_CLASS: &str = "ss";

thread_local! {
    static SS_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

const SS_OUTPUT_SYS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State-space model object.",
}];
const SS_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State matrix with shape n-by-n.",
};
const SS_PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix with shape n-by-nu.",
};
const SS_PARAM_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output matrix with shape ny-by-n.",
};
const SS_PARAM_D: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Feedthrough matrix with shape ny-by-nu.",
};
const SS_INPUTS_ABCD: [BuiltinParamDescriptor; 4] =
    [SS_PARAM_A, SS_PARAM_B, SS_PARAM_C, SS_PARAM_D];
const SS_INPUTS_ABCD_TS: [BuiltinParamDescriptor; 5] = [
    SS_PARAM_A,
    SS_PARAM_B,
    SS_PARAM_C,
    SS_PARAM_D,
    BuiltinParamDescriptor {
        name: "Ts",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0.0"),
        description: "Sample time (0 for continuous-time model).",
    },
];
const SS_INPUTS_ABCD_NAMEVALUE: [BuiltinParamDescriptor; 6] = [
    SS_PARAM_A,
    SS_PARAM_B,
    SS_PARAM_C,
    SS_PARAM_D,
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option name ('Ts' or 'SampleTime').",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value.",
    },
];
const SS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "sys = ss(A, B, C, D)",
        inputs: &SS_INPUTS_ABCD,
        outputs: &SS_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = ss(A, B, C, D, Ts)",
        inputs: &SS_INPUTS_ABCD_TS,
        outputs: &SS_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = ss(A, B, C, D, \"Ts\", Ts)",
        inputs: &SS_INPUTS_ABCD_NAMEVALUE,
        outputs: &SS_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = ss(A, B, C, D, name, value, ...)",
        inputs: &SS_INPUTS_ABCD_NAMEVALUE,
        outputs: &SS_OUTPUT_SYS,
    },
];
const SS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INVALID_ARGUMENT",
    identifier: Some("RunMat:ss:InvalidArgument"),
    when: "Arguments do not match supported ss invocation forms.",
    message: "ss: invalid argument",
};
const SS_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INVALID_OPTION",
    identifier: Some("RunMat:ss:InvalidOption"),
    when: "A name/value option token is unsupported or malformed.",
    message: "ss: invalid option",
};
const SS_ERROR_INVALID_SAMPLE_TIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INVALID_SAMPLE_TIME",
    identifier: Some("RunMat:ss:InvalidSampleTime"),
    when: "Sample time is not a finite non-negative scalar.",
    message: "ss: sample time must be a finite non-negative scalar",
};
const SS_ERROR_INVALID_DIMENSIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INVALID_DIMENSIONS",
    identifier: Some("RunMat:ss:InvalidDimensions"),
    when: "A, B, C, and D dimensions do not define a consistent state-space model.",
    message: "ss: invalid state-space matrix dimensions",
};
const SS_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:ss:UnsupportedInput"),
    when: "An input is complex, sparse, logical, or another unsupported model form.",
    message: "ss: unsupported input",
};
const SS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INTERNAL",
    identifier: Some("RunMat:ss:Internal"),
    when: "Internal tensor/object construction failed.",
    message: "ss: internal error",
};
const SS_ERRORS: [BuiltinErrorDescriptor; 6] = [
    SS_ERROR_INVALID_ARGUMENT,
    SS_ERROR_INVALID_OPTION,
    SS_ERROR_INVALID_SAMPLE_TIME,
    SS_ERROR_INVALID_DIMENSIONS,
    SS_ERROR_UNSUPPORTED_INPUT,
    SS_ERROR_INTERNAL,
];
pub const SS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SS_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::ss")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ss",
    op_kind: GpuOpKind::Custom("state-space-model-constructor"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Object construction runs on the host. gpuArray matrix inputs are gathered before validating and storing the state-space metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::ss")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ss",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "State-space construction is metadata-only and terminates numeric fusion chains.",
};

fn ss_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ss_error_with_message(error.message, error)
}

fn ss_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    ss_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn ss_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ensure_ss_class_registered() {
    SS_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        let mut properties = HashMap::new();
        for name in [
            "A",
            "B",
            "C",
            "D",
            "Ts",
            "InputDelay",
            "OutputDelay",
            "StateName",
            "InputName",
            "OutputName",
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

        let methods: HashMap<String, MethodDef> = HashMap::new();
        runmat_builtins::register_class(ClassDef {
            name: SS_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
        registered.set(true);
    });
}

#[runtime_builtin(
    name = "ss",
    category = "control",
    summary = "Create state-space model objects from A, B, C, and D matrices.",
    keywords = "ss,state space,control system,model,matrices",
    type_resolver(ss_type),
    descriptor(crate::builtins::control::ss::SS_DESCRIPTOR),
    builtin_path = "crate::builtins::control::ss"
)]
async fn ss_builtin(
    a: Value,
    b: Value,
    c: Value,
    d: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let options = SsOptions::parse(&rest)?;
    let a = RealMatrix::parse("A", a).await?;
    let b = RealMatrix::parse("B", b).await?;
    let c = RealMatrix::parse("C", c).await?;
    let d = RealMatrix::parse("D", d).await?;

    validate_state_space_dimensions(&a, &b, &c, &d)?;

    let state_count = a.rows;
    let input_count = b.cols;
    let output_count = c.rows;

    ensure_ss_class_registered();
    let mut object = ObjectInstance::new(SS_CLASS.to_string());
    object.properties.insert("A".to_string(), a.into_value());
    object.properties.insert("B".to_string(), b.into_value());
    object.properties.insert("C".to_string(), c.into_value());
    object.properties.insert("D".to_string(), d.into_value());
    object
        .properties
        .insert("Ts".to_string(), Value::Num(options.sample_time));
    object.properties.insert(
        "InputDelay".to_string(),
        zero_tensor_value(vec![input_count, 1])?,
    );
    object.properties.insert(
        "OutputDelay".to_string(),
        zero_tensor_value(vec![output_count, 1])?,
    );
    object.properties.insert(
        "StateName".to_string(),
        empty_name_cell_value(state_count, 1)?,
    );
    object.properties.insert(
        "InputName".to_string(),
        empty_name_cell_value(input_count, 1)?,
    );
    object.properties.insert(
        "OutputName".to_string(),
        empty_name_cell_value(output_count, 1)?,
    );
    Ok(Value::Object(object))
}

#[derive(Clone)]
struct SsOptions {
    sample_time: f64,
}

impl SsOptions {
    fn parse(rest: &[Value]) -> BuiltinResult<Self> {
        let mut options = Self { sample_time: 0.0 };

        match rest {
            [] => {}
            [sample_time] => options.sample_time = parse_sample_time(sample_time)?,
            _ => {
                if !rest.len().is_multiple_of(2) {
                    return Err(ss_error_with_detail(
                        &SS_ERROR_INVALID_ARGUMENT,
                        "optional arguments must be name-value pairs or a scalar sample time",
                    ));
                }
                let mut idx = 0;
                while idx < rest.len() {
                    let name = scalar_text(&rest[idx], "option name")?;
                    let lowered = name.trim().to_ascii_lowercase();
                    let value = &rest[idx + 1];
                    match lowered.as_str() {
                        "ts" | "sampletime" => options.sample_time = parse_sample_time(value)?,
                        _ => {
                            return Err(ss_error_with_detail(
                                &SS_ERROR_INVALID_OPTION,
                                format!("unsupported option '{name}'"),
                            ));
                        }
                    }
                    idx += 2;
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
            return Err(ss_error_with_detail(
                &SS_ERROR_INVALID_SAMPLE_TIME,
                format!("expected non-negative scalar, got {other:?}"),
            ))
        }
    };
    if !sample_time.is_finite() || sample_time < 0.0 {
        return Err(ss_error(&SS_ERROR_INVALID_SAMPLE_TIME));
    }
    Ok(sample_time)
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        other => Err(ss_error_with_detail(
            &SS_ERROR_INVALID_ARGUMENT,
            format!("{context} must be a string scalar or character vector, got {other:?}"),
        )),
    }
}

#[derive(Clone)]
struct RealMatrix {
    tensor: Tensor,
    rows: usize,
    cols: usize,
}

impl RealMatrix {
    async fn parse(label: &str, value: Value) -> BuiltinResult<Self> {
        let gathered = dispatcher::gather_if_needed_async(&value).await?;
        let tensor = match gathered {
            Value::Tensor(tensor) => tensor,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|err| {
                ss_error_with_detail(&SS_ERROR_INTERNAL, format!("failed to build tensor: {err}"))
            })?,
            Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|err| {
                ss_error_with_detail(&SS_ERROR_INTERNAL, format!("failed to build tensor: {err}"))
            })?,
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                return Err(ss_error_with_detail(
                    &SS_ERROR_UNSUPPORTED_INPUT,
                    format!(
                        "{label} must be finite real numeric data; complex input is unsupported"
                    ),
                ));
            }
            other => {
                return Err(ss_error_with_detail(
                    &SS_ERROR_UNSUPPORTED_INPUT,
                    format!("{label} must be a finite real numeric matrix, got {other:?}"),
                ));
            }
        };

        if tensor.shape.len() > 2 {
            return Err(ss_error_with_detail(
                &SS_ERROR_INVALID_DIMENSIONS,
                format!("{label} must be a 2-D matrix, got shape {:?}", tensor.shape),
            ));
        }
        if tensor.data.iter().any(|value| !value.is_finite()) {
            return Err(ss_error_with_detail(
                &SS_ERROR_UNSUPPORTED_INPUT,
                format!("{label} must contain only finite real values"),
            ));
        }

        Ok(Self {
            rows: tensor.rows,
            cols: tensor.cols,
            tensor,
        })
    }

    fn into_value(self) -> Value {
        Value::Tensor(self.tensor)
    }
}

fn validate_state_space_dimensions(
    a: &RealMatrix,
    b: &RealMatrix,
    c: &RealMatrix,
    d: &RealMatrix,
) -> BuiltinResult<()> {
    if a.rows != a.cols {
        return Err(ss_error_with_detail(
            &SS_ERROR_INVALID_DIMENSIONS,
            format!("A must be square, got {}x{}", a.rows, a.cols),
        ));
    }

    let state_count = a.rows;
    if b.rows != state_count {
        return Err(ss_error_with_detail(
            &SS_ERROR_INVALID_DIMENSIONS,
            format!(
                "B must have {} rows to match A, got {}x{}",
                state_count, b.rows, b.cols
            ),
        ));
    }
    if c.cols != state_count {
        return Err(ss_error_with_detail(
            &SS_ERROR_INVALID_DIMENSIONS,
            format!(
                "C must have {} columns to match A, got {}x{}",
                state_count, c.rows, c.cols
            ),
        ));
    }
    if d.rows != c.rows || d.cols != b.cols {
        return Err(ss_error_with_detail(
            &SS_ERROR_INVALID_DIMENSIONS,
            format!(
                "D must have shape {}x{} to match C outputs and B inputs, got {}x{}",
                c.rows, b.cols, d.rows, d.cols
            ),
        ));
    }

    Ok(())
}

fn zero_tensor_value(shape: Vec<usize>) -> BuiltinResult<Value> {
    let len = shape.iter().product();
    Tensor::new(vec![0.0; len], shape)
        .map(Value::Tensor)
        .map_err(|err| {
            ss_error_with_detail(&SS_ERROR_INTERNAL, format!("failed to build tensor: {err}"))
        })
}

fn empty_name_cell_value(rows: usize, cols: usize) -> BuiltinResult<Value> {
    let len = rows * cols;
    let values = (0..len)
        .map(|_| Value::CharArray(CharArray::new_row("")))
        .collect();
    CellArray::new(values, rows, cols)
        .map(Value::Cell)
        .map_err(|err| {
            ss_error_with_detail(
                &SS_ERROR_INTERNAL,
                format!("failed to build cell array: {err}"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::IntValue;

    fn run_ss(a: Value, b: Value, c: Value, d: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(ss_builtin(a, b, c, d, rest))
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

    fn assert_tensor(value: &Value, shape: &[usize], data: &[f64]) {
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, shape);
                assert_eq!(tensor.data, data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn ss_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"sys = ss(A, B, C, D)"));
        assert!(labels.contains(&"sys = ss(A, B, C, D, Ts)"));
        assert!(labels.contains(&"sys = ss(A, B, C, D, \"Ts\", Ts)"));
        assert!(labels.contains(&"sys = ss(A, B, C, D, name, value, ...)"));
    }

    #[test]
    fn ss_constructs_continuous_state_space_object() {
        let sys = run_ss(
            Value::Tensor(Tensor::new(vec![0.0, -2.0, 1.0, -3.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Num(0.0),
            Vec::new(),
        )
        .expect("ss");

        let Value::Object(object) = &sys else {
            panic!("expected object");
        };
        assert_eq!(object.class_name, "ss");
        assert_eq!(property(&sys, "Ts"), &Value::Num(0.0));
        assert_tensor(property(&sys, "A"), &[2, 2], &[0.0, -2.0, 1.0, -3.0]);
        assert_tensor(property(&sys, "B"), &[2, 1], &[0.0, 1.0]);
        assert_tensor(property(&sys, "C"), &[1, 2], &[1.0, 0.0]);
        assert_tensor(property(&sys, "D"), &[1, 1], &[0.0]);
        assert_tensor(property(&sys, "InputDelay"), &[1, 1], &[0.0]);
        assert_tensor(property(&sys, "OutputDelay"), &[1, 1], &[0.0]);
    }

    #[test]
    fn ss_preserves_matrix_orientation_for_mimo_systems() {
        let sys = run_ss(
            Value::Num(-1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.1, 0.2, 0.3], vec![2, 2]).unwrap()),
            Vec::new(),
        )
        .expect("ss");

        assert_tensor(property(&sys, "A"), &[1, 1], &[-1.0]);
        assert_tensor(property(&sys, "B"), &[1, 2], &[1.0, 2.0]);
        assert_tensor(property(&sys, "C"), &[2, 1], &[3.0, 4.0]);
        assert_tensor(property(&sys, "D"), &[2, 2], &[0.0, 0.1, 0.2, 0.3]);
        assert_tensor(property(&sys, "InputDelay"), &[2, 1], &[0.0, 0.0]);
        assert_tensor(property(&sys, "OutputDelay"), &[2, 1], &[0.0, 0.0]);
    }

    #[test]
    fn ss_accepts_discrete_sample_time() {
        let sys = run_ss(
            Value::Int(IntValue::I32(1)),
            Value::Int(IntValue::I32(2)),
            Value::Int(IntValue::I32(3)),
            Value::Int(IntValue::I32(4)),
            vec![Value::Num(0.25)],
        )
        .expect("ss");

        assert_eq!(property(&sys, "Ts"), &Value::Num(0.25));
    }

    #[test]
    fn ss_accepts_sample_time_name_value_options() {
        let sys = run_ss(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
            vec![Value::from("SampleTime"), Value::Num(0.5)],
        )
        .expect("ss");

        assert_eq!(property(&sys, "Ts"), &Value::Num(0.5));
    }

    #[test]
    fn ss_rejects_nonsquare_a_matrix() {
        let err = run_ss(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .expect_err("nonsquare A should fail");
        assert!(err.message().contains("A must be square"));
        assert_eq!(err.identifier(), SS_ERROR_INVALID_DIMENSIONS.identifier);
    }

    #[test]
    fn ss_rejects_b_row_mismatch() {
        let err = run_ss(
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .expect_err("B mismatch should fail");
        assert!(err.message().contains("B must have 2 rows"));
        assert_eq!(err.identifier(), SS_ERROR_INVALID_DIMENSIONS.identifier);
    }

    #[test]
    fn ss_rejects_d_shape_mismatch() {
        let err = run_ss(
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .expect_err("D mismatch should fail");
        assert!(err.message().contains("D must have shape 1x1"));
        assert_eq!(err.identifier(), SS_ERROR_INVALID_DIMENSIONS.identifier);
    }

    #[test]
    fn ss_rejects_invalid_sample_time() {
        let err = run_ss(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(0.0),
            vec![Value::Num(-0.1)],
        )
        .expect_err("negative Ts should fail");
        assert_eq!(err.identifier(), SS_ERROR_INVALID_SAMPLE_TIME.identifier);
    }

    #[test]
    fn ss_rejects_complex_inputs() {
        let err = run_ss(
            Value::Complex(1.0, 1.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(0.0),
            Vec::new(),
        )
        .expect_err("complex A should fail");
        assert!(err.message().contains("complex input is unsupported"));
        assert_eq!(err.identifier(), SS_ERROR_UNSUPPORTED_INPUT.identifier);
    }

    #[test]
    fn ss_gpu_matrix_input_gathers_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let sys = run_ss(
                Value::GpuTensor(handle),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
                Vec::new(),
            )
            .expect("ss");

            assert_tensor(property(&sys, "A"), &[1, 1], &[1.0]);
        });
    }
}

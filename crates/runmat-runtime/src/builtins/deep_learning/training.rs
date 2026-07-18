use runmat_builtins::{NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::BuiltinResult;

use super::{
    any_type, deep_learning_error, gather_args, object, parse_name_values, scalar_text,
    text_or_missing, unsupported_error,
};

#[runtime_builtin(
    name = "trainingOptions",
    category = "deep_learning",
    summary = "Create training options compatibility objects.",
    keywords = "trainingOptions,deep learning,training,options,solver",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn training_options_builtin(
    solver: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let solver = scalar_text(&solver, "trainingOptions")?.to_ascii_lowercase();
    match solver.as_str() {
        "sgdm" | "adam" | "rmsprop" => {}
        other => {
            return Err(deep_learning_error(
                "trainingOptions",
                format!("trainingOptions: unsupported solver '{other}'"),
            ));
        }
    }
    let mut properties = vec![
        ("SolverName".to_string(), Value::String(solver.clone())),
        ("MaxEpochs".to_string(), Value::Num(30.0)),
        ("MiniBatchSize".to_string(), Value::Num(128.0)),
        (
            "InitialLearnRate".to_string(),
            Value::Num(if solver == "adam" { 0.001 } else { 0.01 }),
        ),
        ("Shuffle".to_string(), Value::String("once".into())),
        ("Verbose".to_string(), Value::Bool(true)),
        ("Plots".to_string(), Value::String("none".into())),
        (
            "ExecutionEnvironment".to_string(),
            Value::String("auto".into()),
        ),
    ];
    for (key, value) in parse_name_values(gather_args(rest).await?, "trainingOptions")? {
        properties.push((canonical_training_option(&key), value));
    }
    Ok(object("nnet.cnn.TrainingOptions", properties))
}

fn canonical_training_option(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "maxepochs" => "MaxEpochs",
        "minibatchsize" => "MiniBatchSize",
        "initiallearnrate" => "InitialLearnRate",
        "shuffle" => "Shuffle",
        "verbose" => "Verbose",
        "plots" => "Plots",
        "executionenvironment" => "ExecutionEnvironment",
        "validationdata" => "ValidationData",
        "validationfrequency" => "ValidationFrequency",
        "learnrateschedule" => "LearnRateSchedule",
        "learnratedropperiod" => "LearnRateDropPeriod",
        "learnratedropfactor" => "LearnRateDropFactor",
        "gradientthreshold" => "GradientThreshold",
        "l2regularization" => "L2Regularization",
        "momentum" => "Momentum",
        "squaredgradientdecayfactor" => "SquaredGradientDecayFactor",
        "gradientdecayfactor" => "GradientDecayFactor",
        "epsilon" => "Epsilon",
        other => other,
    }
    .to_string()
}

#[runtime_builtin(
    name = "trainNetwork",
    category = "deep_learning",
    summary = "Report that full neural-network training is not yet implemented.",
    keywords = "trainNetwork,deep learning,training,network",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn train_network_builtin(_args: Vec<Value>) -> BuiltinResult<Value> {
    Err(unsupported_error(
        "trainNetwork",
        "trainNetwork requires optimizer, autodiff, datastore, and layer execution infrastructure; layer/options compatibility objects are implemented in this slice",
    ))
}

#[runtime_builtin(
    name = "trainnet",
    category = "deep_learning",
    summary = "Report that full neural-network training is not yet implemented.",
    keywords = "trainnet,deep learning,training,network",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn trainnet_builtin(_args: Vec<Value>) -> BuiltinResult<Value> {
    Err(unsupported_error(
        "trainnet",
        "trainnet requires optimizer, autodiff, datastore, and layer execution infrastructure; layer/options compatibility objects are implemented in this slice",
    ))
}

#[runtime_builtin(
    name = "dlarray",
    category = "deep_learning",
    summary = "Create a dlarray compatibility object around host numeric data.",
    keywords = "dlarray,deep learning,array,labels",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlarray_builtin(data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(rest).await?;
    if gathered.len() > 1 {
        return Err(deep_learning_error(
            "dlarray",
            "dlarray: only the optional dimension-label format argument is supported",
        ));
    }
    if matches!(data, Value::GpuTensor(_)) {
        return Err(deep_learning_error(
            "dlarray",
            "dlarray: gpuArray-backed dlarray values require deep-learning GPU array semantics not implemented in this slice",
        ));
    }
    let format = text_or_missing(gathered.first(), "", "dlarray")?;
    Ok(object(
        "dlarray",
        vec![
            ("Data", data),
            ("Format", Value::String(format.clone())),
            ("Labels", Value::String(format)),
        ],
    ))
}

#[runtime_builtin(
    name = "dlfeval",
    category = "deep_learning",
    summary = "Evaluate a function handle in Deep Learning compatibility context.",
    keywords = "dlfeval,deep learning,autodiff,function evaluation",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::DLFEVAL_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlfeval_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let Some((function, rest)) = args.split_first() else {
        return Err(deep_learning_error(
            "dlfeval",
            "dlfeval: expected a function handle followed by input arguments",
        ));
    };
    if !matches!(
        function,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    ) {
        return Err(deep_learning_error(
            "dlfeval",
            format!("dlfeval: expected a function handle, got {function:?}"),
        ));
    }
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);
    crate::call_feval_async_with_outputs(function.clone(), rest, requested_outputs).await
}

#[runtime_builtin(
    name = "adamupdate",
    category = "deep_learning",
    summary = "Update numeric parameters using the Adam optimizer rule.",
    keywords = "adamupdate,deep learning,adam,optimizer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ADAMUPDATE_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn adamupdate_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let options = AdamUpdateArgs::parse(args)?;
    let eval = evaluate_adamupdate(options)?;
    match crate::output_count::current_output_count() {
        None => Ok(eval.parameters),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![eval.parameters])),
        Some(2) => Ok(Value::OutputList(vec![eval.parameters, eval.average_grad])),
        Some(3) => Ok(Value::OutputList(vec![
            eval.parameters,
            eval.average_grad,
            eval.average_sq_grad,
        ])),
        Some(requested) => Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: requested {requested} outputs, but at most 3 are supported"),
        )),
    }
}

#[runtime_builtin(
    name = "dlupdate",
    category = "deep_learning",
    summary = "Report that model parameter tree updates are not yet implemented.",
    keywords = "dlupdate,deep learning,model update",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlupdate_builtin(_args: Vec<Value>) -> BuiltinResult<Value> {
    Err(unsupported_error(
        "dlupdate",
        "dlupdate requires model parameter tree traversal and function-handle invocation semantics",
    ))
}

#[runtime_builtin(
    name = "exportONNXNetwork",
    category = "deep_learning",
    summary = "Report that ONNX export is not yet implemented.",
    keywords = "exportONNXNetwork,deep learning,onnx,export",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn export_onnx_network_builtin(_args: Vec<Value>) -> BuiltinResult<Value> {
    Err(unsupported_error(
        "exportONNXNetwork",
        "exportONNXNetwork requires ONNX graph serialization and trained-network execution metadata",
    ))
}

struct AdamUpdateArgs {
    parameters: NumericPayload,
    gradient: NumericPayload,
    average_grad: NumericPayload,
    average_sq_grad: NumericPayload,
    iteration: usize,
    learn_rate: f64,
    gradient_decay_factor: f64,
    squared_gradient_decay_factor: f64,
    epsilon: f64,
}

impl AdamUpdateArgs {
    fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        if !(5..=9).contains(&args.len()) {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: expected 5 to 9 positional arguments",
            ));
        }
        let parameters = NumericPayload::parse(&args[0], "parameters", true)?;
        if parameters.data.is_empty() {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: parameters must not be empty",
            ));
        }
        let gradient = NumericPayload::parse(&args[1], "grad", true)?;
        let average_grad = NumericPayload::parse(&args[2], "averageGrad", false)?
            .with_default_shape(&parameters)?;
        let average_sq_grad = NumericPayload::parse(&args[3], "averageSqGrad", false)?
            .with_default_shape(&parameters)?;
        require_same_shape(&parameters, &gradient, "grad")?;
        require_same_shape(&parameters, &average_grad, "averageGrad")?;
        require_same_shape(&parameters, &average_sq_grad, "averageSqGrad")?;
        let iteration = positive_iteration(&args[4])?;
        Ok(Self {
            parameters,
            gradient,
            average_grad,
            average_sq_grad,
            iteration,
            learn_rate: optional_positive_scalar(&args, 5, 0.001, "learnRate")?,
            gradient_decay_factor: optional_unit_scalar(&args, 6, 0.9, "gradDecay")?,
            squared_gradient_decay_factor: optional_unit_scalar(&args, 7, 0.999, "sqGradDecay")?,
            epsilon: optional_positive_scalar(&args, 8, 1.0e-8, "epsilon")?,
        })
    }
}

struct AdamUpdateEval {
    parameters: Value,
    average_grad: Value,
    average_sq_grad: Value,
}

fn evaluate_adamupdate(args: AdamUpdateArgs) -> BuiltinResult<AdamUpdateEval> {
    let count = args.parameters.data.len();
    let mut updated_parameters = Vec::with_capacity(count);
    let mut updated_average_grad = Vec::with_capacity(count);
    let mut updated_average_sq_grad = Vec::with_capacity(count);
    let iteration = args.iteration as f64;
    let grad_correction = 1.0 - args.gradient_decay_factor.powf(iteration);
    let sq_grad_correction = 1.0 - args.squared_gradient_decay_factor.powf(iteration);
    if grad_correction <= 0.0 || sq_grad_correction <= 0.0 {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: decay factors and iteration produced invalid bias correction",
        ));
    }
    for idx in 0..count {
        let grad = args.gradient.data[idx];
        let avg_grad = args.gradient_decay_factor * args.average_grad.data[idx]
            + (1.0 - args.gradient_decay_factor) * grad;
        let avg_sq_grad = args.squared_gradient_decay_factor * args.average_sq_grad.data[idx]
            + (1.0 - args.squared_gradient_decay_factor) * grad * grad;
        let corrected_grad = avg_grad / grad_correction;
        let corrected_sq_grad = avg_sq_grad / sq_grad_correction;
        let step = args.learn_rate * corrected_grad / (corrected_sq_grad.sqrt() + args.epsilon);
        let updated = args.parameters.data[idx] - step;
        if !updated.is_finite() || !avg_grad.is_finite() || !avg_sq_grad.is_finite() {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: update produced a non-finite value",
            ));
        }
        updated_parameters.push(updated);
        updated_average_grad.push(avg_grad);
        updated_average_sq_grad.push(avg_sq_grad);
    }
    Ok(AdamUpdateEval {
        parameters: args.parameters.materialize(updated_parameters)?,
        average_grad: args.average_grad.materialize(updated_average_grad)?,
        average_sq_grad: args.average_sq_grad.materialize(updated_average_sq_grad)?,
    })
}

#[derive(Clone)]
struct NumericPayload {
    data: Vec<f64>,
    shape: Vec<usize>,
    repr: NumericRepr,
}

impl NumericPayload {
    fn parse(value: &Value, label: &'static str, reject_empty: bool) -> BuiltinResult<Self> {
        let payload = match value {
            Value::Num(n) if n.is_finite() => Self {
                data: vec![*n],
                shape: vec![1, 1],
                repr: NumericRepr::Scalar,
            },
            Value::Int(i) => Self {
                data: vec![i.to_f64()],
                shape: vec![1, 1],
                repr: NumericRepr::Scalar,
            },
            Value::Tensor(tensor) => {
                if !matches!(tensor.dtype, NumericDType::F64 | NumericDType::F32) {
                    return Err(deep_learning_error(
                        "adamupdate",
                        format!(
                            "adamupdate: {label} tensor must be double or single, got {}",
                            tensor.dtype.class_name()
                        ),
                    ));
                }
                ensure_finite(&tensor.data, label)?;
                Self {
                    data: tensor.data.clone(),
                    shape: tensor.shape.clone(),
                    repr: NumericRepr::Tensor {
                        shape: tensor.shape.clone(),
                        dtype: tensor.dtype,
                    },
                }
            }
            Value::Object(object) if object.class_name == "dlarray" => {
                let data = object.properties.get("Data").ok_or_else(|| {
                    deep_learning_error("adamupdate", "adamupdate: dlarray is missing Data")
                })?;
                let inner = Self::parse(data, label, reject_empty)?;
                Self {
                    data: inner.data,
                    shape: inner.shape,
                    repr: NumericRepr::Dlarray {
                        inner: Box::new(inner.repr),
                        format: object
                            .properties
                            .get("Format")
                            .cloned()
                            .unwrap_or_else(|| Value::String(String::new())),
                        labels: object
                            .properties
                            .get("Labels")
                            .cloned()
                            .unwrap_or_else(|| Value::String(String::new())),
                    },
                }
            }
            Value::GpuTensor(_) => {
                return Err(deep_learning_error(
                    "adamupdate",
                    "adamupdate: gpuArray optimizer updates require provider kernels and are not implemented in this runtime slice",
                ));
            }
            other => {
                return Err(deep_learning_error(
                    "adamupdate",
                    format!("adamupdate: {label} must be a finite numeric array or dlarray, got {other:?}"),
                ));
            }
        };
        if reject_empty && payload.data.is_empty() {
            return Err(deep_learning_error(
                "adamupdate",
                format!("adamupdate: {label} must not be empty"),
            ));
        }
        Ok(payload)
    }

    fn with_default_shape(mut self, fallback: &NumericPayload) -> BuiltinResult<Self> {
        if !self.data.is_empty() {
            return Ok(self);
        }
        self.data = vec![0.0; fallback.data.len()];
        self.shape = fallback.shape.clone();
        self.repr = fallback.repr.clone();
        Ok(self)
    }

    fn materialize(&self, data: Vec<f64>) -> BuiltinResult<Value> {
        self.repr.materialize(data)
    }
}

#[derive(Clone)]
enum NumericRepr {
    Scalar,
    Tensor {
        shape: Vec<usize>,
        dtype: NumericDType,
    },
    Dlarray {
        inner: Box<NumericRepr>,
        format: Value,
        labels: Value,
    },
}

impl NumericRepr {
    fn materialize(&self, data: Vec<f64>) -> BuiltinResult<Value> {
        match self {
            Self::Scalar => match data.as_slice() {
                [value] => Ok(Value::Num(*value)),
                _ => {
                    let len = data.len();
                    Tensor::new(data, vec![1, len])
                        .map(Value::Tensor)
                        .map_err(|err| deep_learning_error("adamupdate", err))
                }
            },
            Self::Tensor { shape, dtype } => Tensor::new_with_dtype(data, shape.clone(), *dtype)
                .map(Value::Tensor)
                .map_err(|err| deep_learning_error("adamupdate", err)),
            Self::Dlarray {
                inner,
                format,
                labels,
            } => Ok(object(
                "dlarray",
                vec![
                    ("Data", inner.materialize(data)?),
                    ("Format", format.clone()),
                    ("Labels", labels.clone()),
                ],
            )),
        }
    }
}

fn ensure_finite(values: &[f64], label: &'static str) -> BuiltinResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must contain finite values"),
        ));
    }
    Ok(())
}

fn require_same_shape(
    expected: &NumericPayload,
    actual: &NumericPayload,
    label: &'static str,
) -> BuiltinResult<()> {
    if expected.shape != actual.shape || expected.data.len() != actual.data.len() {
        return Err(deep_learning_error(
            "adamupdate",
            format!(
                "adamupdate: {label} must match parameter shape {:?}, got {:?}",
                expected.shape, actual.shape
            ),
        ));
    }
    Ok(())
}

fn positive_iteration(value: &Value) -> BuiltinResult<usize> {
    let iteration = optional_positive_scalar(std::slice::from_ref(value), 0, 0.0, "iteration")?;
    if iteration.fract().abs() > f64::EPSILON || iteration > usize::MAX as f64 {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: iteration must be a positive integer",
        ));
    }
    Ok(iteration as usize)
}

fn optional_finite_scalar(
    args: &[Value],
    index: usize,
    default: f64,
    label: &'static str,
) -> BuiltinResult<f64> {
    let Some(value) = args.get(index) else {
        return Ok(default);
    };
    match value {
        Value::Num(n) if n.is_finite() => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) if tensor.data.len() == 1 && tensor.data[0].is_finite() => {
            Ok(tensor.data[0])
        }
        other => Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be a finite numeric scalar, got {other:?}"),
        )),
    }
}

fn optional_positive_scalar(
    args: &[Value],
    index: usize,
    default: f64,
    label: &'static str,
) -> BuiltinResult<f64> {
    let value = optional_finite_scalar(args, index, default, label)?;
    if value <= 0.0 {
        return Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be positive"),
        ));
    }
    Ok(value)
}

fn optional_unit_scalar(
    args: &[Value],
    index: usize,
    default: f64,
    label: &'static str,
) -> BuiltinResult<f64> {
    let value = optional_finite_scalar(args, index, default, label)?;
    if !(0.0..1.0).contains(&value) {
        return Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be greater than or equal to 0 and less than 1"),
        ));
    }
    Ok(value)
}

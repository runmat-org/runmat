use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::BuiltinResult;

use super::{
    any_type, deep_learning_error, gather_args, numeric_values, object, parse_name_values,
    scalar_text, text_or_missing, unsupported_error,
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
    summary = "Report that automatic differentiation function evaluation is not yet implemented.",
    keywords = "dlfeval,deep learning,autodiff,function evaluation",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlfeval_builtin(_args: Vec<Value>) -> BuiltinResult<Value> {
    Err(unsupported_error(
        "dlfeval",
        "dlfeval requires automatic differentiation and function-handle invocation over traced dlarray values",
    ))
}

#[runtime_builtin(
    name = "adamupdate",
    category = "deep_learning",
    summary = "Report that Adam optimizer state updates are not yet implemented.",
    keywords = "adamupdate,deep learning,adam,optimizer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn adamupdate_builtin(_args: Vec<Value>) -> BuiltinResult<Value> {
    Err(unsupported_error(
        "adamupdate",
        "adamupdate requires optimizer-state semantics over dlarray/model gradients and remains in the training/autodiff slice",
    ))
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
    name = "crossentropy",
    category = "deep_learning",
    summary = "Compute cross-entropy loss for numeric prediction and target arrays.",
    keywords = "crossentropy,deep learning,loss",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn crossentropy_builtin(
    predictions: Value,
    targets: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: optional format/reduction arguments are not implemented in this compatibility slice",
        ));
    }
    let y = finite_numeric_values(&predictions, "crossentropy", "predictions")?;
    let t = finite_numeric_values(&targets, "crossentropy", "targets")?;
    if y.len() != t.len() {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: predictions and targets must have the same number of elements",
        ));
    }
    if y.is_empty() {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: predictions and targets must not be empty",
        ));
    }
    let mut loss = 0.0;
    let eps = 1.0e-12;
    for (yp, target) in y.iter().zip(t.iter()) {
        let clipped = yp.clamp(eps, 1.0 - eps);
        loss -= target * clipped.ln();
    }
    Ok(Value::Num(loss / y.len() as f64))
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

fn finite_numeric_values(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    let values = numeric_values(value, function, label)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(deep_learning_error(
            function,
            format!("{function}: {label} must contain finite numeric values"),
        ));
    }
    Ok(values)
}

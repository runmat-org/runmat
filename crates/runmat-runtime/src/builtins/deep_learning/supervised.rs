use std::collections::BTreeMap;

use runmat_builtins::{ObjectInstance, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{builtins::common::tensor, BuiltinResult};

use super::{
    any_type, deep_learning_error, gather_args, model, numeric_values, positive_usize, scalar_text,
    string_array,
};

#[runtime_builtin(
    name = "trainNetwork",
    category = "deep_learning",
    summary = "Train a supported feed-forward Deep Learning network.",
    keywords = "trainNetwork,deep learning,training,network",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::supervised"
)]
pub(super) async fn train_network_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args(args).await?;
    if args.len() != 4 {
        return Err(deep_learning_error(
            "trainNetwork",
            "trainNetwork: expected X, Y, layers, and trainingOptions",
        ));
    }
    let x = numeric_matrix(args[0].clone(), "trainNetwork", "X")?;
    let mut layers = model::layers_from_network_value(args[2].clone(), "trainNetwork")?;
    model::validate_forward_layers(&layers, "trainNetwork")?;
    model::initialise_fully_connected_layers(&mut layers, "trainNetwork")?;
    let config = TrainingConfig::from_value(&args[3], "trainNetwork")?;
    let network_class = if matches!(
        &args[2],
        Value::Object(object) if object.class_name == "nnet.cnn.LayerGraph"
    ) {
        model::DAG_NETWORK_CLASS
    } else {
        model::SERIES_NETWORK_CLASS
    };
    let task = infer_train_network_task(&layers, &args[1], x.rows)?;
    train_and_output(layers, x, task, config, network_class, "trainNetwork")
}

#[runtime_builtin(
    name = "trainnet",
    category = "deep_learning",
    summary = "Train a supported dlnetwork with a built-in loss.",
    keywords = "trainnet,deep learning,training,network",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::supervised"
)]
pub(super) async fn trainnet_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args(args).await?;
    if args.len() != 5 {
        return Err(deep_learning_error(
            "trainnet",
            "trainnet: expected X, Y, net, lossFcn, and trainingOptions",
        ));
    }
    let x = numeric_matrix(args[0].clone(), "trainnet", "X")?;
    let mut layers = model::layers_from_network_value(args[2].clone(), "trainnet")?;
    model::validate_forward_layers(&layers, "trainnet")?;
    model::initialise_fully_connected_layers(&mut layers, "trainnet")?;
    let loss = parse_trainnet_loss(&args[3])?;
    let config = TrainingConfig::from_value(&args[4], "trainnet")?;
    let task = task_from_loss(loss, &args[1], x.rows)?;
    train_and_output(layers, x, task, config, model::DLNETWORK_CLASS, "trainnet")
}

fn train_and_output(
    mut layers: Vec<Value>,
    x: Matrix,
    task: Task,
    config: TrainingConfig,
    network_class: &str,
    function: &'static str,
) -> BuiltinResult<Value> {
    let mut state = OptimizerState::new(config.solver);
    let mut losses = Vec::with_capacity(config.max_epochs);
    let mut iteration = 0usize;
    let mut order: Vec<usize> = (0..x.rows).collect();
    if matches!(config.shuffle, ShuffleMode::Once) {
        order.reverse();
    }

    for epoch in 0..config.max_epochs {
        if matches!(config.shuffle, ShuffleMode::EveryEpoch) && !order.is_empty() {
            let amount = epoch % order.len();
            order.rotate_left(amount);
        }
        let mut epoch_loss = 0.0;
        let mut seen = 0usize;
        for chunk in order.chunks(config.mini_batch_size) {
            iteration += 1;
            let batch_x = x.select_rows(chunk, function)?;
            let batch_targets = task.select_rows(chunk, function)?;
            let (loss, grad, caches) =
                loss_and_gradient(&layers, &batch_x, &batch_targets, function)?;
            let batch_len = chunk.len();
            epoch_loss += loss * batch_len as f64;
            seen += batch_len;
            backward_update(
                &mut layers,
                caches,
                grad,
                &config,
                &mut state,
                iteration,
                function,
            )?;
        }
        let averaged = if seen == 0 {
            0.0
        } else {
            epoch_loss / seen as f64
        };
        losses.push(averaged);
    }

    let mut network = model::network_object(network_class, layers, BTreeMap::new(), function)?;
    attach_training_metadata(&mut network, &task, &config, &losses, iteration, function)?;
    output_network(
        network,
        training_info(&config, &losses, iteration, function)?,
        function,
    )
}

fn output_network(network: Value, info: Value, function: &'static str) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        None => Ok(network),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![network])),
        Some(2) => Ok(Value::OutputList(vec![network, info])),
        Some(requested) => Err(deep_learning_error(
            function,
            format!("{function}: requested {requested} outputs, but at most 2 are supported"),
        )),
    }
}

fn attach_training_metadata(
    network: &mut Value,
    task: &Task,
    config: &TrainingConfig,
    losses: &[f64],
    iteration: usize,
    function: &'static str,
) -> BuiltinResult<()> {
    let Value::Object(object) = network else {
        return Ok(());
    };
    object.properties.insert(
        "TrainingInfo".to_string(),
        training_info(config, losses, iteration, function)?,
    );
    object.properties.insert(
        "TrainedWith".to_string(),
        Value::String(config.solver.as_str().into()),
    );
    object.properties.insert(
        "TrainingLoss".to_string(),
        Value::String(task.loss_name().to_string()),
    );
    if let Task::Classification(classification) = task {
        object.properties.insert(
            "Classes".to_string(),
            classification.classes_value(function)?,
        );
    }
    Ok(())
}

fn training_info(
    config: &TrainingConfig,
    losses: &[f64],
    iteration: usize,
    function: &'static str,
) -> BuiltinResult<Value> {
    let mut info = StructValue::new();
    info.insert("SolverName", Value::String(config.solver.as_str().into()));
    info.insert("MaxEpochs", Value::Num(config.max_epochs as f64));
    info.insert("MiniBatchSize", Value::Num(config.mini_batch_size as f64));
    info.insert("InitialLearnRate", Value::Num(config.learn_rate));
    info.insert("Iteration", Value::Num(iteration as f64));
    info.insert("Epoch", Value::Num(config.max_epochs as f64));
    info.insert("FinalLoss", Value::Num(*losses.last().unwrap_or(&0.0)));
    info.insert(
        "TrainingLoss",
        Value::Tensor(
            Tensor::new(losses.to_vec(), vec![losses.len(), 1])
                .map_err(|err| deep_learning_error(function, err))?,
        ),
    );
    info.insert(
        "StopReason",
        Value::String("MaxEpochs completed".to_string()),
    );
    Ok(Value::Struct(info))
}

#[derive(Clone)]
struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn from_tensor(tensor: Tensor, function: &'static str, label: &str) -> BuiltinResult<Self> {
        if tensor.shape.len() > 2 {
            return Err(deep_learning_error(
                function,
                format!("{function}: {label} must be a 2-D numeric matrix"),
            ));
        }
        let rows = tensor.rows;
        let cols = tensor.cols;
        let data = tensor::tensor_into_values_f64(tensor);
        if data.iter().any(|value| !value.is_finite()) {
            return Err(deep_learning_error(
                function,
                format!("{function}: {label} must contain finite numeric values"),
            ));
        }
        Ok(Self { data, rows, cols })
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row + col * self.rows] = value;
    }

    fn select_rows(&self, rows: &[usize], function: &'static str) -> BuiltinResult<Self> {
        let mut out = vec![0.0; rows.len() * self.cols];
        for col in 0..self.cols {
            for (out_row, &row) in rows.iter().enumerate() {
                if row >= self.rows {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: internal mini-batch row index out of range"),
                    ));
                }
                out[out_row + col * rows.len()] = self.get(row, col);
            }
        }
        Ok(Self {
            data: out,
            rows: rows.len(),
            cols: self.cols,
        })
    }

    fn into_tensor(self, function: &'static str) -> BuiltinResult<Tensor> {
        Tensor::new(self.data, vec![self.rows, self.cols])
            .map_err(|err| deep_learning_error(function, err))
    }
}

fn numeric_matrix(value: Value, function: &'static str, label: &str) -> BuiltinResult<Matrix> {
    let tensor = crate::builtins::common::tensor::value_into_tensor_for(function, value)
        .map_err(|err| deep_learning_error(function, format!("{function}: {err}")))?;
    Matrix::from_tensor(tensor, function, label)
}

#[derive(Clone)]
enum Task {
    Regression(Matrix),
    Classification(ClassificationTarget),
}

impl Task {
    fn select_rows(&self, rows: &[usize], function: &'static str) -> BuiltinResult<Self> {
        match self {
            Self::Regression(targets) => targets.select_rows(rows, function).map(Self::Regression),
            Self::Classification(targets) => {
                Ok(Self::Classification(targets.select_rows(rows, function)?))
            }
        }
    }

    fn loss_name(&self) -> &'static str {
        match self {
            Self::Regression(_) => "mse",
            Self::Classification(_) => "crossentropy",
        }
    }
}

#[derive(Clone)]
struct ClassificationTarget {
    classes: Vec<String>,
    one_hot: Matrix,
}

impl ClassificationTarget {
    fn select_rows(&self, rows: &[usize], function: &'static str) -> BuiltinResult<Self> {
        Ok(Self {
            classes: self.classes.clone(),
            one_hot: self.one_hot.select_rows(rows, function)?,
        })
    }

    fn classes_value(&self, function: &'static str) -> BuiltinResult<Value> {
        string_array(self.classes.clone(), vec![1, self.classes.len()], function)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LossKind {
    Regression,
    Classification,
}

fn infer_train_network_task(
    layers: &[Value],
    y: &Value,
    observations: usize,
) -> BuiltinResult<Task> {
    if layers.iter().any(|layer| {
        matches!(
            layer,
            Value::Object(object) if object.class_name == "nnet.cnn.layer.ClassificationOutputLayer"
        )
    }) {
        classification_target(y, observations, "trainNetwork").map(Task::Classification)
    } else if layers.iter().any(|layer| {
        matches!(
            layer,
            Value::Object(object) if object.class_name == "nnet.cnn.layer.RegressionOutputLayer"
        )
    }) {
        numeric_matrix(y.clone(), "trainNetwork", "Y").and_then(|target| {
            validate_target_rows(&target, observations, "trainNetwork")?;
            Ok(Task::Regression(target))
        })
    } else if is_label_target(y) {
        classification_target(y, observations, "trainNetwork").map(Task::Classification)
    } else {
        numeric_matrix(y.clone(), "trainNetwork", "Y").and_then(|target| {
            validate_target_rows(&target, observations, "trainNetwork")?;
            Ok(Task::Regression(target))
        })
    }
}

fn task_from_loss(loss: LossKind, y: &Value, observations: usize) -> BuiltinResult<Task> {
    match loss {
        LossKind::Regression => numeric_matrix(y.clone(), "trainnet", "Y").and_then(|target| {
            validate_target_rows(&target, observations, "trainnet")?;
            Ok(Task::Regression(target))
        }),
        LossKind::Classification => {
            classification_target(y, observations, "trainnet").map(Task::Classification)
        }
    }
}

fn parse_trainnet_loss(value: &Value) -> BuiltinResult<LossKind> {
    if is_function_handle(value) {
        return Err(deep_learning_error(
            "trainnet",
            "trainnet: custom loss function handles require autodiff infrastructure and are not supported in this training slice",
        ));
    }
    match scalar_text(value, "trainnet")?
        .to_ascii_lowercase()
        .as_str()
    {
        "mse" | "mean-squared-error" | "mean_squared_error" => Ok(LossKind::Regression),
        "crossentropy" | "cross-entropy" => Ok(LossKind::Classification),
        other => Err(deep_learning_error(
            "trainnet",
            format!("trainnet: unsupported loss function '{other}'"),
        )),
    }
}

fn is_function_handle(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}

fn validate_target_rows(
    target: &Matrix,
    observations: usize,
    function: &'static str,
) -> BuiltinResult<()> {
    if target.rows != observations {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: Y must have one row per observation; expected {observations}, got {}",
                target.rows
            ),
        ));
    }
    Ok(())
}

fn is_label_target(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => true,
        Value::Cell(cell) => cell.data.iter().all(is_label_target),
        Value::Object(object) => object.class_name == "categorical",
        _ => false,
    }
}

fn classification_target(
    value: &Value,
    observations: usize,
    function: &'static str,
) -> BuiltinResult<ClassificationTarget> {
    if let Some(labels) = label_strings(value, function)? {
        if labels.len() != observations {
            return Err(deep_learning_error(
                function,
                format!(
                    "{function}: classification labels must have one label per observation; expected {observations}, got {}",
                    labels.len()
                ),
            ));
        }
        return one_hot_from_labels(labels, function);
    }
    let target = numeric_matrix(value.clone(), function, "Y")?;
    validate_target_rows(&target, observations, function)?;
    if target.cols > 1 {
        let classes = (1..=target.cols)
            .map(|idx| idx.to_string())
            .collect::<Vec<_>>();
        return Ok(ClassificationTarget {
            classes,
            one_hot: target,
        });
    }
    let labels = target
        .data
        .iter()
        .map(|value| {
            if !value.is_finite() || *value < 1.0 || value.fract().abs() > f64::EPSILON {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: numeric classification labels must be positive integers"),
                ));
            }
            Ok(format!("{}", *value as usize))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    one_hot_from_labels(labels, function)
}

fn label_strings(value: &Value, function: &'static str) -> BuiltinResult<Option<Vec<String>>> {
    match value {
        Value::String(text) => Ok(Some(vec![text.clone()])),
        Value::CharArray(chars) if chars.rows == 1 => Ok(Some(vec![chars.data.iter().collect()])),
        Value::StringArray(array) => Ok(Some(array.data.clone())),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for item in &cell.data {
                labels.push(scalar_text(item, function)?);
            }
            Ok(Some(labels))
        }
        Value::Object(object) if object.class_name == "categorical" => {
            let categories = match object.properties.get("Categories") {
                Some(Value::StringArray(categories)) => categories.data.clone(),
                _ => {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: categorical labels are missing Categories"),
                    ));
                }
            };
            let codes = match object.properties.get("Codes") {
                Some(Value::Tensor(codes)) => codes,
                _ => {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: categorical labels are missing numeric Codes"),
                    ));
                }
            };
            let mut labels = Vec::with_capacity(codes.data.len());
            for code in &codes.data {
                if code.is_nan() {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: classification labels must not be missing"),
                    ));
                }
                if *code < 1.0 || code.fract().abs() > f64::EPSILON {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: categorical Codes must be positive integers"),
                    ));
                }
                let idx = *code as usize - 1;
                let Some(label) = categories.get(idx) else {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: categorical Codes reference an unknown category"),
                    ));
                };
                labels.push(label.clone());
            }
            Ok(Some(labels))
        }
        _ => Ok(None),
    }
}

fn one_hot_from_labels(
    labels: Vec<String>,
    function: &'static str,
) -> BuiltinResult<ClassificationTarget> {
    let mut classes = labels.clone();
    classes.sort();
    classes.dedup();
    if classes.is_empty() {
        return Err(deep_learning_error(
            function,
            format!("{function}: classification labels must not be empty"),
        ));
    }
    let mut matrix = Matrix::zeros(labels.len(), classes.len());
    for (row, label) in labels.iter().enumerate() {
        let class = classes.binary_search(label).map_err(|_| {
            deep_learning_error(function, "internal classification class lookup failed")
        })?;
        matrix.set(row, class, 1.0);
    }
    Ok(ClassificationTarget {
        classes,
        one_hot: matrix,
    })
}

#[derive(Clone, Copy)]
enum Solver {
    Sgdm,
    Adam,
}

impl Solver {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sgdm => "sgdm",
            Self::Adam => "adam",
        }
    }
}

#[derive(Clone, Copy)]
enum ShuffleMode {
    Never,
    Once,
    EveryEpoch,
}

struct TrainingConfig {
    solver: Solver,
    max_epochs: usize,
    mini_batch_size: usize,
    learn_rate: f64,
    momentum: f64,
    gradient_decay_factor: f64,
    squared_gradient_decay_factor: f64,
    epsilon: f64,
    shuffle: ShuffleMode,
}

impl TrainingConfig {
    fn from_value(value: &Value, function: &'static str) -> BuiltinResult<Self> {
        let Value::Object(object) = value else {
            return Err(deep_learning_error(
                function,
                format!("{function}: expected a trainingOptions object"),
            ));
        };
        if object.class_name != "nnet.cnn.TrainingOptions" {
            return Err(deep_learning_error(
                function,
                format!("{function}: expected a trainingOptions object"),
            ));
        }
        let solver_name = text_property(object, "SolverName", "sgdm", function)?;
        let solver = match solver_name.to_ascii_lowercase().as_str() {
            "sgdm" => Solver::Sgdm,
            "adam" => Solver::Adam,
            "rmsprop" => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: RMSProp training is not supported in this slice"),
                ));
            }
            other => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: unsupported solver '{other}'"),
                ));
            }
        };
        let max_epochs = positive_option_usize(object, "MaxEpochs", 30, function)?;
        let mini_batch_size = positive_option_usize(object, "MiniBatchSize", 128, function)?;
        let learn_rate = positive_option_f64(object, "InitialLearnRate", 0.01, function)?;
        validate_execution_environment(object, function)?;
        Ok(Self {
            solver,
            max_epochs,
            mini_batch_size,
            learn_rate,
            momentum: unit_option_f64(object, "Momentum", 0.9, function)?,
            gradient_decay_factor: unit_option_f64(object, "GradientDecayFactor", 0.9, function)?,
            squared_gradient_decay_factor: unit_option_f64(
                object,
                "SquaredGradientDecayFactor",
                0.999,
                function,
            )?,
            epsilon: positive_option_f64(object, "Epsilon", 1.0e-8, function)?,
            shuffle: shuffle_mode(
                &text_property(object, "Shuffle", "once", function)?,
                function,
            )?,
        })
    }
}

fn validate_execution_environment(
    object: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<()> {
    let environment = text_property(object, "ExecutionEnvironment", "auto", function)?;
    match environment.to_ascii_lowercase().as_str() {
        "auto" | "cpu" => Ok(()),
        "gpu" | "multi-gpu" | "multigpu" | "parallel" => Err(deep_learning_error(
            function,
            format!(
                "{function}: ExecutionEnvironment '{environment}' requires native GPU or parallel training support"
            ),
        )),
        other => Err(deep_learning_error(
            function,
            format!("{function}: unsupported ExecutionEnvironment value '{other}'"),
        )),
    }
}

fn text_property(
    object: &ObjectInstance,
    name: &str,
    default: &str,
    function: &'static str,
) -> BuiltinResult<String> {
    match object.properties.get(name) {
        Some(value) => scalar_text(value, function),
        None => Ok(default.to_string()),
    }
}

fn positive_option_usize(
    object: &ObjectInstance,
    name: &str,
    default: usize,
    function: &'static str,
) -> BuiltinResult<usize> {
    match object.properties.get(name) {
        Some(value) => positive_usize(value, function, name).map_err(|_| {
            deep_learning_error(
                function,
                format!("{function}: {name} must be a positive integer scalar"),
            )
        }),
        None => Ok(default),
    }
}

fn positive_option_f64(
    object: &ObjectInstance,
    name: &str,
    default: f64,
    function: &'static str,
) -> BuiltinResult<f64> {
    match object.properties.get(name) {
        Some(value) => {
            let values = numeric_values(value, function, name)?;
            if values.len() != 1 || !values[0].is_finite() || values[0] <= 0.0 {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: {name} must be a positive numeric scalar"),
                ));
            }
            Ok(values[0])
        }
        None => Ok(default),
    }
}

fn unit_option_f64(
    object: &ObjectInstance,
    name: &str,
    default: f64,
    function: &'static str,
) -> BuiltinResult<f64> {
    match object.properties.get(name) {
        Some(value) => {
            let values = numeric_values(value, function, name)?;
            if values.len() != 1 || !values[0].is_finite() || values[0] < 0.0 || values[0] >= 1.0 {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: {name} must be greater than or equal to 0 and less than 1"
                    ),
                ));
            }
            Ok(values[0])
        }
        None => Ok(default),
    }
}

fn shuffle_mode(value: &str, function: &'static str) -> BuiltinResult<ShuffleMode> {
    match value.to_ascii_lowercase().as_str() {
        "never" => Ok(ShuffleMode::Never),
        "once" => Ok(ShuffleMode::Once),
        "every-epoch" | "everyepoch" => Ok(ShuffleMode::EveryEpoch),
        other => Err(deep_learning_error(
            function,
            format!("{function}: unsupported Shuffle value '{other}'"),
        )),
    }
}

fn loss_and_gradient(
    layers: &[Value],
    x: &Matrix,
    targets: &Task,
    function: &'static str,
) -> BuiltinResult<(f64, Matrix, Vec<LayerCache>)> {
    let (prediction, caches) = forward_training(layers, x, function)?;
    match targets {
        Task::Regression(target) => {
            if prediction.rows != target.rows || prediction.cols != target.cols {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: regression target size must match network output size {:?}",
                        [prediction.rows, prediction.cols]
                    ),
                ));
            }
            let mut loss = 0.0;
            let mut grad = Matrix::zeros(prediction.rows, prediction.cols);
            for idx in 0..prediction.data.len() {
                let diff = prediction.data[idx] - target.data[idx];
                loss += 0.5 * diff * diff;
                grad.data[idx] = diff / prediction.rows as f64;
            }
            Ok((loss / prediction.rows as f64, grad, caches))
        }
        Task::Classification(target) => {
            if prediction.rows != target.one_hot.rows || prediction.cols != target.one_hot.cols {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: classification target size must match network output size {:?}",
                        [prediction.rows, prediction.cols]
                    ),
                ));
            }
            let has_softmax = layers.iter().any(|layer| {
                matches!(
                    layer,
                    Value::Object(object) if object.class_name == "nnet.cnn.layer.SoftmaxLayer"
                )
            });
            let probabilities = if has_softmax {
                prediction.clone()
            } else {
                softmax(&prediction, function)?
            };
            let eps = 1.0e-12;
            let mut loss = 0.0;
            let mut grad = Matrix::zeros(probabilities.rows, probabilities.cols);
            if has_softmax {
                for idx in 0..probabilities.data.len() {
                    let probability = probabilities.data[idx].clamp(eps, 1.0);
                    let target_value = target.one_hot.data[idx];
                    loss -= target_value * probability.ln();
                    grad.data[idx] = if target_value == 0.0 {
                        0.0
                    } else {
                        -target_value / probability / probabilities.rows as f64
                    };
                }
            } else {
                for idx in 0..probabilities.data.len() {
                    let probability = probabilities.data[idx].clamp(eps, 1.0);
                    let target_value = target.one_hot.data[idx];
                    loss -= target_value * probability.ln();
                    grad.data[idx] =
                        (probabilities.data[idx] - target_value) / probabilities.rows as f64;
                }
            }
            Ok((loss / probabilities.rows as f64, grad, caches))
        }
    }
}

enum LayerCache {
    FullyConnected {
        layer_index: usize,
        input: Matrix,
        weights: Matrix,
    },
    Relu {
        input: Matrix,
    },
    Elu {
        input: Matrix,
        alpha: f64,
    },
    Softmax {
        output: Matrix,
    },
}

fn forward_training(
    layers: &[Value],
    input: &Matrix,
    function: &'static str,
) -> BuiltinResult<(Matrix, Vec<LayerCache>)> {
    let mut current = input.clone();
    let mut caches = Vec::new();
    let mut saw_input = false;
    for (layer_index, layer) in layers.iter().enumerate() {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        current = match layer.class_name.as_str() {
            "nnet.cnn.layer.FeatureInputLayer" => {
                let expected = model::feature_input_width(layer, function)?;
                if current.cols != expected {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: input has {} features, but featureInputLayer expects {expected}",
                            current.cols
                        ),
                    ));
                }
                saw_input = true;
                current
            }
            "nnet.cnn.layer.FullyConnectedLayer" => {
                let weights = Matrix::from_tensor(
                    model::tensor_property(layer, "Weights", function)?,
                    function,
                    "Weights",
                )?;
                let bias = Matrix::from_tensor(
                    model::tensor_property(layer, "Bias", function)?,
                    function,
                    "Bias",
                )?;
                let output = fully_connected(&current, &weights, &bias, function)?;
                caches.push(LayerCache::FullyConnected {
                    layer_index,
                    input: current,
                    weights,
                });
                output
            }
            "nnet.cnn.layer.ReLULayer" => {
                let previous = current;
                let output = map_matrix(&previous, |value| value.max(0.0));
                caches.push(LayerCache::Relu { input: previous });
                output
            }
            "nnet.cnn.layer.ELULayer" => {
                let alpha = layer
                    .properties
                    .get("Alpha")
                    .map(|value| numeric_values(value, function, "Alpha"))
                    .transpose()?
                    .and_then(|values| values.first().copied())
                    .unwrap_or(1.0);
                let previous = current;
                let output = map_matrix(&previous, |value| {
                    if value > 0.0 {
                        value
                    } else {
                        alpha * (value.exp() - 1.0)
                    }
                });
                caches.push(LayerCache::Elu {
                    input: previous,
                    alpha,
                });
                output
            }
            "nnet.cnn.layer.SoftmaxLayer" => {
                let output = softmax(&current, function)?;
                caches.push(LayerCache::Softmax {
                    output: output.clone(),
                });
                output
            }
            "nnet.cnn.layer.ClassificationOutputLayer" | "nnet.cnn.layer.RegressionOutputLayer" => {
                current
            }
            other => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: unsupported layer type '{other}'"),
                ));
            }
        };
    }
    if !saw_input {
        return Err(deep_learning_error(
            function,
            format!("{function}: network must start with a supported input layer"),
        ));
    }
    Ok((current, caches))
}

fn fully_connected(
    input: &Matrix,
    weights: &Matrix,
    bias: &Matrix,
    function: &'static str,
) -> BuiltinResult<Matrix> {
    if weights.cols != input.cols {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Weights must be outputSize-by-{}",
                input.cols
            ),
        ));
    }
    if bias.data.len() != weights.rows {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: fullyConnectedLayer Bias must have {} elements",
                weights.rows
            ),
        ));
    }
    let mut out = Matrix::zeros(input.rows, weights.rows);
    for row in 0..input.rows {
        for out_col in 0..weights.rows {
            let mut acc = bias.data[out_col];
            for feature in 0..input.cols {
                acc += input.get(row, feature) * weights.get(out_col, feature);
            }
            out.set(row, out_col, acc);
        }
    }
    Ok(out)
}

fn map_matrix(input: &Matrix, f: impl Fn(f64) -> f64) -> Matrix {
    Matrix {
        data: input.data.iter().map(|value| f(*value)).collect(),
        rows: input.rows,
        cols: input.cols,
    }
}

fn softmax(input: &Matrix, function: &'static str) -> BuiltinResult<Matrix> {
    let mut out = Matrix::zeros(input.rows, input.cols);
    for row in 0..input.rows {
        let max_value = (0..input.cols)
            .map(|col| input.get(row, col))
            .fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0;
        for col in 0..input.cols {
            let value = (input.get(row, col) - max_value).exp();
            out.set(row, col, value);
            denom += value;
        }
        if !denom.is_finite() || denom <= 0.0 {
            return Err(deep_learning_error(
                function,
                format!("{function}: softmax produced invalid normalization"),
            ));
        }
        for col in 0..input.cols {
            out.set(row, col, out.get(row, col) / denom);
        }
    }
    Ok(out)
}

fn backward_update(
    layers: &mut [Value],
    caches: Vec<LayerCache>,
    mut grad: Matrix,
    config: &TrainingConfig,
    state: &mut OptimizerState,
    iteration: usize,
    function: &'static str,
) -> BuiltinResult<()> {
    for cache in caches.into_iter().rev() {
        grad = match cache {
            LayerCache::Softmax { output } => softmax_backward(&output, &grad),
            LayerCache::Relu { input } => relu_backward(&input, &grad),
            LayerCache::Elu { input, alpha } => elu_backward(&input, &grad, alpha),
            LayerCache::FullyConnected {
                layer_index,
                input,
                weights,
            } => {
                let (grad_input, grad_weights, grad_bias) =
                    fully_connected_backward(&input, &weights, &grad);
                update_fully_connected_layer(
                    &mut layers[layer_index],
                    grad_weights,
                    grad_bias,
                    config,
                    state,
                    iteration,
                    layer_index,
                    function,
                )?;
                grad_input
            }
        };
    }
    Ok(())
}

fn softmax_backward(output: &Matrix, grad: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(output.rows, output.cols);
    for row in 0..output.rows {
        let mut dot = 0.0;
        for col in 0..output.cols {
            dot += grad.get(row, col) * output.get(row, col);
        }
        for col in 0..output.cols {
            out.set(row, col, output.get(row, col) * (grad.get(row, col) - dot));
        }
    }
    out
}

fn relu_backward(input: &Matrix, grad: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(input.rows, input.cols);
    for idx in 0..input.data.len() {
        out.data[idx] = if input.data[idx] > 0.0 {
            grad.data[idx]
        } else {
            0.0
        };
    }
    out
}

fn elu_backward(input: &Matrix, grad: &Matrix, alpha: f64) -> Matrix {
    let mut out = Matrix::zeros(input.rows, input.cols);
    for idx in 0..input.data.len() {
        out.data[idx] = if input.data[idx] > 0.0 {
            grad.data[idx]
        } else {
            grad.data[idx] * alpha * input.data[idx].exp()
        };
    }
    out
}

fn fully_connected_backward(
    input: &Matrix,
    weights: &Matrix,
    grad: &Matrix,
) -> (Matrix, Matrix, Matrix) {
    let mut grad_input = Matrix::zeros(input.rows, input.cols);
    let mut grad_weights = Matrix::zeros(weights.rows, weights.cols);
    let mut grad_bias = Matrix::zeros(weights.rows, 1);

    for row in 0..input.rows {
        for out_col in 0..weights.rows {
            let g = grad.get(row, out_col);
            grad_bias.data[out_col] += g;
            for feature in 0..input.cols {
                grad_weights.data[out_col + feature * weights.rows] += input.get(row, feature) * g;
                grad_input.data[row + feature * input.rows] += g * weights.get(out_col, feature);
            }
        }
    }
    (grad_input, grad_weights, grad_bias)
}

fn update_fully_connected_layer(
    layer: &mut Value,
    grad_weights: Matrix,
    grad_bias: Matrix,
    config: &TrainingConfig,
    state: &mut OptimizerState,
    iteration: usize,
    layer_index: usize,
    function: &'static str,
) -> BuiltinResult<()> {
    let Value::Object(object) = layer else {
        return Err(deep_learning_error(
            function,
            format!("{function}: network layers must be layer objects"),
        ));
    };
    let mut weights = Matrix::from_tensor(
        model::tensor_property(object, "Weights", function)?,
        function,
        "Weights",
    )?;
    let mut bias = Matrix::from_tensor(
        model::tensor_property(object, "Bias", function)?,
        function,
        "Bias",
    )?;
    state.update_parameter(
        &mut weights.data,
        &grad_weights.data,
        config,
        iteration,
        layer_index,
        "Weights",
        function,
    )?;
    state.update_parameter(
        &mut bias.data,
        &grad_bias.data,
        config,
        iteration,
        layer_index,
        "Bias",
        function,
    )?;
    object.properties.insert(
        "Weights".to_string(),
        Value::Tensor(weights.into_tensor(function)?),
    );
    object.properties.insert(
        "Bias".to_string(),
        Value::Tensor(bias.into_tensor(function)?),
    );
    Ok(())
}

struct OptimizerState {
    solver: Solver,
    slots: BTreeMap<(usize, &'static str), OptimizerSlot>,
}

impl OptimizerState {
    fn new(solver: Solver) -> Self {
        Self {
            solver,
            slots: BTreeMap::new(),
        }
    }

    fn update_parameter(
        &mut self,
        values: &mut [f64],
        grad: &[f64],
        config: &TrainingConfig,
        iteration: usize,
        layer_index: usize,
        parameter: &'static str,
        function: &'static str,
    ) -> BuiltinResult<()> {
        if values.len() != grad.len() {
            return Err(deep_learning_error(
                function,
                format!("{function}: optimizer gradient shape does not match {parameter}"),
            ));
        }
        let slot = self
            .slots
            .entry((layer_index, parameter))
            .or_insert_with(|| OptimizerSlot::new(values.len()));
        match self.solver {
            Solver::Sgdm => {
                for idx in 0..values.len() {
                    slot.m[idx] = config.momentum * slot.m[idx] - config.learn_rate * grad[idx];
                    values[idx] += slot.m[idx];
                }
            }
            Solver::Adam => {
                let t = iteration as f64;
                let beta1 = config.gradient_decay_factor;
                let beta2 = config.squared_gradient_decay_factor;
                let m_correction = 1.0 - beta1.powf(t);
                let v_correction = 1.0 - beta2.powf(t);
                if m_correction <= 0.0 || v_correction <= 0.0 {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: Adam bias correction is invalid"),
                    ));
                }
                for idx in 0..values.len() {
                    slot.m[idx] = beta1 * slot.m[idx] + (1.0 - beta1) * grad[idx];
                    slot.v[idx] = beta2 * slot.v[idx] + (1.0 - beta2) * grad[idx] * grad[idx];
                    let m_hat = slot.m[idx] / m_correction;
                    let v_hat = slot.v[idx] / v_correction;
                    values[idx] -= config.learn_rate * m_hat / (v_hat.sqrt() + config.epsilon);
                }
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(deep_learning_error(
                function,
                format!("{function}: optimizer produced non-finite {parameter} values"),
            ));
        }
        Ok(())
    }
}

struct OptimizerSlot {
    m: Vec<f64>,
    v: Vec<f64>,
}

impl OptimizerSlot {
    fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }
}

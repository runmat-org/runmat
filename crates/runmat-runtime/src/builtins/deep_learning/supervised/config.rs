use super::*;

#[derive(Clone, Copy)]
pub(super) enum Solver {
    Sgdm,
    Adam,
}

impl Solver {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Sgdm => "sgdm",
            Self::Adam => "adam",
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum ShuffleMode {
    Never,
    Once,
    EveryEpoch,
}

pub(super) struct TrainingConfig {
    pub(super) solver: Solver,
    pub(super) max_epochs: usize,
    pub(super) mini_batch_size: usize,
    pub(super) learn_rate: f64,
    pub(super) momentum: f64,
    pub(super) gradient_decay_factor: f64,
    pub(super) squared_gradient_decay_factor: f64,
    pub(super) epsilon: f64,
    pub(super) shuffle: ShuffleMode,
}

impl TrainingConfig {
    pub(super) fn from_value(value: &Value, function: &'static str) -> BuiltinResult<Self> {
        let Value::Object(object) = value else {
            return Err(deep_learning_error(
                function,
                format!("{function}: expected a trainingOptions object"),
            ));
        };
        if !object.class_name.is(super::super::TRAINING_OPTIONS_CLASS) {
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

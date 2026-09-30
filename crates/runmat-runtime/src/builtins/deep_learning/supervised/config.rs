use super::*;
use runmat_builtins::TrainingOptionKey as O;
use std::collections::HashMap;

#[cfg(test)]
mod tests;

struct TrainingOptionsView<'a> {
    values: HashMap<O, &'a Value>,
}

impl<'a> TrainingOptionsView<'a> {
    fn parse(object: &'a ObjectInstance, function: &'static str) -> BuiltinResult<Self> {
        let mut values = HashMap::with_capacity(object.properties.len());
        for (name, value) in &object.properties {
            let key = O::from_name(name).ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: unsupported training option '{name}'"),
                )
            })?;
            if values.insert(key, value).is_some() {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: duplicate training option {}",
                        key.canonical_name()
                    ),
                ));
            }
        }
        Ok(Self { values })
    }

    fn get(&self, key: O) -> Option<&'a Value> {
        self.values.get(&key).copied()
    }

    fn admit_unsupported(&self, function: &'static str) -> BuiltinResult<()> {
        for (&key, value) in &self.values {
            let admitted = match key {
                O::SolverName
                | O::MaxEpochs
                | O::MiniBatchSize
                | O::InitialLearnRate
                | O::Shuffle
                | O::ExecutionEnvironment
                | O::Momentum
                | O::SquaredGradientDecayFactor
                | O::GradientDecayFactor
                | O::Epsilon => true,
                O::Verbose => {
                    let _ = super::super::logical_scalar(value, function, "Verbose")?;
                    true
                }
                O::Plots | O::LearnRateSchedule => {
                    scalar_text(value, function)?.eq_ignore_ascii_case("none")
                }
                O::L2Regularization => {
                    let values = numeric_values(value, function, key.canonical_name())?;
                    values.len() == 1 && values[0] == 0.0
                }
                O::ValidationData
                | O::ValidationFrequency
                | O::LearnRateDropPeriod
                | O::LearnRateDropFactor
                | O::GradientThreshold => false,
            };
            if !admitted {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: training option {} is not supported by this execution route",
                        key.canonical_name()
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
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
        let options = TrainingOptionsView::parse(object, function)?;
        options.admit_unsupported(function)?;
        let solver_name = text_property(&options, O::SolverName, "sgdm", function)?;
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
        if solver == Solver::Adam && options.get(O::Momentum).is_some() {
            return Err(deep_learning_error(
                function,
                format!("{function}: Momentum is not consumed by Adam training"),
            ));
        }
        if solver == Solver::Sgdm
            && [
                O::GradientDecayFactor,
                O::SquaredGradientDecayFactor,
                O::Epsilon,
            ]
            .iter()
            .any(|&key| options.get(key).is_some())
        {
            return Err(deep_learning_error(
                function,
                format!("{function}: Adam controls are not consumed by SGDM training"),
            ));
        }
        let max_epochs = positive_option_usize(&options, O::MaxEpochs, 30, function)?;
        let mini_batch_size = positive_option_usize(&options, O::MiniBatchSize, 128, function)?;
        let learn_rate = positive_option_f64(&options, O::InitialLearnRate, 0.01, function)?;
        validate_execution_environment(&options, function)?;
        Ok(Self {
            solver,
            max_epochs,
            mini_batch_size,
            learn_rate,
            momentum: unit_option_f64(&options, O::Momentum, 0.9, function)?,
            gradient_decay_factor: unit_option_f64(
                &options,
                O::GradientDecayFactor,
                0.9,
                function,
            )?,
            squared_gradient_decay_factor: unit_option_f64(
                &options,
                O::SquaredGradientDecayFactor,
                0.999,
                function,
            )?,
            epsilon: positive_option_f64(&options, O::Epsilon, 1.0e-8, function)?,
            shuffle: shuffle_mode(
                &text_property(&options, O::Shuffle, "once", function)?,
                function,
            )?,
        })
    }
}

fn validate_execution_environment(
    options: &TrainingOptionsView<'_>,
    function: &'static str,
) -> BuiltinResult<()> {
    let environment = text_property(options, O::ExecutionEnvironment, "auto", function)?;
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
    options: &TrainingOptionsView<'_>,
    key: O,
    default: &str,
    function: &'static str,
) -> BuiltinResult<String> {
    match options.get(key) {
        Some(value) => scalar_text(value, function),
        None => Ok(default.to_string()),
    }
}

fn positive_option_usize(
    options: &TrainingOptionsView<'_>,
    key: O,
    default: usize,
    function: &'static str,
) -> BuiltinResult<usize> {
    let name = key.canonical_name();
    match options.get(key) {
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
    options: &TrainingOptionsView<'_>,
    key: O,
    default: f64,
    function: &'static str,
) -> BuiltinResult<f64> {
    let name = key.canonical_name();
    match options.get(key) {
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
    options: &TrainingOptionsView<'_>,
    key: O,
    default: f64,
    function: &'static str,
) -> BuiltinResult<f64> {
    let name = key.canonical_name();
    match options.get(key) {
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

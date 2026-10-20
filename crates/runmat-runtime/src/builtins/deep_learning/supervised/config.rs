use super::*;
use runmat_builtins::TrainingOptionKey as O;
pub(super) use runmat_builtins::TrainingSolver as Solver;
use runmat_value::ObjectInstance;
use std::collections::HashMap;

use crate::builtins::deep_learning::{numeric_values, positive_usize, scalar_text};

#[cfg(test)]
mod tests;

struct TrainingOptionsView<'a> {
    values: HashMap<O, &'a Value>,
}

impl<'a> TrainingOptionsView<'a> {
    fn parse(object: &'a ObjectInstance, function: &'static str) -> BuiltinResult<Self> {
        let mut values = HashMap::with_capacity(object.properties().len());
        for (name, value) in object.properties() {
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
                O::L2Regularization => true,
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
    pub(super) l2_regularization: f64,
    pub(super) shuffle: ShuffleMode,
    pub(super) verbose: bool,
}

impl TrainingConfig {
    pub(super) fn from_value(value: &Value, function: &'static str) -> BuiltinResult<Self> {
        let Value::Object(object) = value else {
            return Err(deep_learning_error(
                function,
                format!("{function}: expected a trainingOptions object"),
            ));
        };
        let solver = Solver::from_identity(object.class_name()).ok_or_else(|| {
            deep_learning_error(
                function,
                format!("{function}: expected a trainingOptions object"),
            )
        })?;
        let options = TrainingOptionsView::parse(object, function)?;
        options.admit_unsupported(function)?;
        if let Some(name) = options.get(O::SolverName) {
            let name = scalar_text(name, function)?;
            if Solver::from_name(&name) != Some(solver) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: SolverName disagrees with the trainingOptions class"),
                ));
            }
        }
        if solver != Solver::Sgdm && options.get(O::Momentum).is_some() {
            return Err(deep_learning_error(
                function,
                format!("{function}: Momentum is consumed only by SGDM training"),
            ));
        }
        if solver != Solver::Adam && options.get(O::GradientDecayFactor).is_some() {
            return Err(deep_learning_error(
                function,
                format!("{function}: GradientDecayFactor is consumed only by Adam training"),
            ));
        }
        if solver == Solver::Sgdm
            && [O::SquaredGradientDecayFactor, O::Epsilon]
                .iter()
                .any(|&key| options.get(key).is_some())
        {
            return Err(deep_learning_error(
                function,
                format!("{function}: adaptive solver controls are not consumed by SGDM training"),
            ));
        }
        let max_epochs = positive_option_usize(&options, O::MaxEpochs, 30, function)?;
        let mini_batch_size = positive_option_usize(&options, O::MiniBatchSize, 128, function)?;
        let learn_rate = positive_option_f64(
            &options,
            O::InitialLearnRate,
            solver.default_initial_learn_rate(),
            function,
        )?;
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
                solver.default_squared_gradient_decay_factor(),
                function,
            )?,
            epsilon: positive_option_f64(&options, O::Epsilon, 1.0e-8, function)?,
            l2_regularization: nonnegative_option_f64(
                &options,
                O::L2Regularization,
                0.0001,
                function,
            )?,
            shuffle: shuffle_mode(
                &text_property(&options, O::Shuffle, "once", function)?,
                function,
            )?,
            verbose: options
                .get(O::Verbose)
                .map(|value| super::super::logical_scalar(value, function, "Verbose"))
                .transpose()?
                .unwrap_or(true),
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

fn nonnegative_option_f64(
    options: &TrainingOptionsView<'_>,
    key: O,
    default: f64,
    function: &'static str,
) -> BuiltinResult<f64> {
    let name = key.canonical_name();
    match options.get(key) {
        Some(value) => {
            let values = numeric_values(value, function, name)?;
            if values.len() != 1 || !values[0].is_finite() || values[0] < 0.0 {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: {name} must be a finite nonnegative scalar"),
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

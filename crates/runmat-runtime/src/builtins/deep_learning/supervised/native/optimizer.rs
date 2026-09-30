//! Native-class supervised parameter steps keyed by canonical Learnables roles.
//! Slot state is private to a training invocation; `Layers` remains the only
//! network parameter authority and receives one atomic selected-role update.

use std::collections::HashMap;

use runmat_accelerate_api::handle_is_explicit;
use runmat_value::{NumericDType, NumericStorage, ObjectInstance, Tensor, Value};

use crate::builtins::deep_learning::{
    deep_learning_error,
    model::{
        parameters::{self, ParameterKey, ParameterUpdate},
        precision::project_owned,
    },
};
use crate::BuiltinResult;

use super::super::{Solver, TrainingConfig};
use super::factors::FactorPlan;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

enum Slot {
    Single { first: Vec<f32>, second: Vec<f32> },
    Double { first: Vec<f64>, second: Vec<f64> },
}

pub(super) struct NativeOptimizer {
    slots: HashMap<ParameterKey, Slot>,
    factors: FactorPlan,
}

impl NativeOptimizer {
    pub(super) fn new(factors: FactorPlan) -> Self {
        Self {
            slots: HashMap::new(),
            factors,
        }
    }

    /// All gradient and source identities, shapes, and classes are admitted
    /// before a step. Source tensors are the native host snapshots used by
    /// the same forward execution; no second provider download occurs here.
    pub(super) fn propose(
        &mut self,
        network: &ObjectInstance,
        prepared_sources: &[(ParameterKey, Tensor)],
        gradients: Vec<(ParameterKey, Tensor)>,
        config: &TrainingConfig,
        iteration: usize,
        function: &'static str,
    ) -> BuiltinResult<Vec<ParameterUpdate>> {
        if iteration == 0 {
            return Err(fail(function, "optimizer iteration must be positive"));
        }
        let rows = parameters::view(network, function)?;
        let mut sources = HashMap::with_capacity(prepared_sources.len());
        for (key, source) in prepared_sources {
            if sources.insert(key, source).is_some() {
                return Err(fail(function, "duplicate prepared parameter source"));
            }
        }
        let mut requested = HashMap::with_capacity(gradients.len());
        for (key, gradient) in gradients {
            if requested.insert(key, gradient).is_some() {
                return Err(fail(function, "duplicate parameter gradient"));
            }
        }
        let mut admitted = Vec::with_capacity(requested.len());
        for row in rows {
            let source = sources
                .remove(&row.key)
                .ok_or_else(|| fail(function, "prepared source differs from Learnables"))?;
            let Some(gradient) = requested.remove(&row.key) else {
                continue;
            };
            let value = row
                .value
                .ok_or_else(|| fail(function, "parameter is uninitialized"))?;
            let shape = parameters::numeric_payload_shape(value, row.key.role.name(), function)?;
            if shape != gradient.shape || gradient.is_empty() {
                return Err(fail(
                    function,
                    "parameter gradient geometry differs from Learnables",
                ));
            }
            let dtype = parameters::floating_precision(value)
                .filter(|dtype| matches!(dtype, NumericDType::F32 | NumericDType::F64))
                .ok_or_else(|| fail(function, "training parameter must be single or double"))?;
            if !matches!(
                gradient.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) {
                return Err(fail(function, "training gradient must be single or double"));
            }
            let data = match value {
                Value::Object(object)
                    if object
                        .class_name
                        .is(crate::builtins::deep_learning::DLARRAY_CLASS) =>
                {
                    object
                        .properties
                        .get("Data")
                        .ok_or_else(|| fail(function, "parameter dlarray is missing Data"))?
                }
                other => other,
            };
            if matches!(data, Value::GpuTensor(handle) if handle_is_explicit(handle)) {
                return Err(fail(
                    function,
                    "explicit gpuArray parameter cannot enter host optimizer",
                ));
            }
            if source.shape.as_slice() != shape || source.numeric_dtype() != dtype {
                return Err(fail(
                    function,
                    "prepared parameter differs from admitted Learnables",
                ));
            }
            admitted.push((row.key, source, gradient, dtype));
        }
        if !requested.is_empty() || !sources.is_empty() {
            return Err(fail(
                function,
                "prepared source or gradient identity differs from Learnables",
            ));
        }
        let mut updates = Vec::with_capacity(admitted.len());
        for (key, value, gradient, dtype) in admitted {
            let factors = self.factors.get(&key, function)?;
            let l2 = config.l2_regularization * factors.l2;
            if !l2.is_finite() || l2 < 0.0 {
                return Err(fail(function, "effective L2 factor is invalid"));
            }
            if value.shape != gradient.shape || value.numeric_dtype() != dtype {
                return Err(fail(function, "prepared parameter changed after admission"));
            }
            let gradient = project_owned(gradient, dtype, function)?;
            let next = match dtype {
                NumericDType::F32 => {
                    let current = value.as_f32_slice().expect("admitted F32");
                    let gradient = gradient.as_f32_slice().expect("projected F32");
                    let slot = self
                        .slots
                        .entry(key.clone())
                        .or_insert_with(|| Slot::Single {
                            first: vec![0.; current.len()],
                            second: if matches!(config.solver, Solver::Adam) {
                                vec![0.; current.len()]
                            } else {
                                Vec::new()
                            },
                        });
                    let Slot::Single { first, second } = slot else {
                        return Err(fail(function, "optimizer slot precision changed"));
                    };
                    NumericStorage::F32(step_values(
                        current,
                        gradient,
                        first,
                        second,
                        config,
                        factors.learning,
                        l2,
                        iteration,
                        function,
                    )?)
                }
                NumericDType::F64 => {
                    let current = value.as_f64_slice().expect("admitted F64");
                    let gradient = gradient.as_f64_slice().expect("projected F64");
                    let slot = self
                        .slots
                        .entry(key.clone())
                        .or_insert_with(|| Slot::Double {
                            first: vec![0.; current.len()],
                            second: if matches!(config.solver, Solver::Adam) {
                                vec![0.; current.len()]
                            } else {
                                Vec::new()
                            },
                        });
                    let Slot::Double { first, second } = slot else {
                        return Err(fail(function, "optimizer slot precision changed"));
                    };
                    NumericStorage::F64(step_values(
                        current,
                        gradient,
                        first,
                        second,
                        config,
                        factors.learning,
                        l2,
                        iteration,
                        function,
                    )?)
                }
                _ => unreachable!("admitted native precision"),
            };
            let value = Tensor::from_numeric_storage(next, value.shape.clone())
                .map_err(|error| fail(function, error))?;
            updates.push(ParameterUpdate { key, value });
        }
        Ok(updates)
    }
}

fn step_values<T: crate::builtins::deep_learning::model::execution::host::numeric::Real>(
    current: &[T],
    gradient: &[T],
    first: &mut [T],
    second: &mut [T],
    config: &TrainingConfig,
    learning_factor: f64,
    l2_coefficient: f64,
    iteration: usize,
    function: &'static str,
) -> BuiltinResult<Vec<T>> {
    if current.len() != gradient.len()
        || current.len() != first.len()
        || (matches!(config.solver, Solver::Adam) && current.len() != second.len())
    {
        return Err(fail(function, "optimizer slot geometry changed"));
    }
    let rate = T::from_f64(config.learn_rate * learning_factor);
    let regularization = T::from_f64(l2_coefficient);
    let momentum = T::from_f64(config.momentum);
    let beta1 = T::from_f64(config.gradient_decay_factor);
    let beta2 = T::from_f64(config.squared_gradient_decay_factor);
    let epsilon = T::from_f64(config.epsilon);
    if !rate.is_finite()
        || rate < T::zero()
        || !regularization.is_finite()
        || regularization < T::zero()
        || !epsilon.is_finite()
        || epsilon <= T::zero()
    {
        return Err(fail(
            function,
            "optimizer controls are not representable in parameter precision",
        ));
    }
    let correction1 = T::from_f64(1.0 - config.gradient_decay_factor.powf(iteration as f64));
    let correction2 =
        T::from_f64(1.0 - config.squared_gradient_decay_factor.powf(iteration as f64));
    if matches!(config.solver, Solver::Adam)
        && (correction1 <= T::zero() || correction2 <= T::zero())
    {
        return Err(fail(function, "Adam bias correction is not representable"));
    }
    let mut next = Vec::new();
    next.try_reserve_exact(current.len())
        .map_err(|_| fail(function, "cannot reserve updated parameter storage"))?;
    for index in 0..current.len() {
        let value = current[index];
        let grad = gradient[index] + regularization * value;
        if !value.is_finite() || !grad.is_finite() {
            return Err(fail(
                function,
                "optimizer requires finite values and gradients",
            ));
        }
        let updated = match config.solver {
            Solver::Sgdm => {
                first[index] = momentum * first[index] - rate * grad;
                value + first[index]
            }
            Solver::Adam => {
                first[index] = beta1 * first[index] + (T::one() - beta1) * grad;
                second[index] = beta2 * second[index] + (T::one() - beta2) * grad * grad;
                let mean = first[index] / correction1;
                let variance = second[index] / correction2;
                value - rate * mean / (variance.sqrt() + epsilon)
            }
        };
        if !updated.is_finite()
            || !first[index].is_finite()
            || (matches!(config.solver, Solver::Adam) && !second[index].is_finite())
        {
            return Err(fail(function, "optimizer step exceeds native precision"));
        }
        next.push(updated);
    }
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::deep_learning::supervised::{ShuffleMode, Solver};

    fn config(solver: Solver) -> TrainingConfig {
        TrainingConfig {
            solver,
            max_epochs: 1,
            mini_batch_size: 1,
            learn_rate: 0.1,
            momentum: 0.9,
            gradient_decay_factor: 0.9,
            squared_gradient_decay_factor: 0.999,
            epsilon: 1.0e-8,
            l2_regularization: 0.0001,
            shuffle: ShuffleMode::Never,
            verbose: false,
        }
    }

    #[test]
    fn l2_regularization_adds_weight_decay_to_native_sgdm_gradient() {
        let mut first = [0.0_f64];
        let mut second = [];
        let updated = step_values(
            &[2.0_f64],
            &[0.0_f64],
            &mut first,
            &mut second,
            &config(Solver::Sgdm),
            1.0,
            0.5,
            1,
            "test",
        )
        .unwrap();
        assert_eq!(updated, vec![1.9]);
        assert_eq!(first, [-0.1]);
    }

    #[test]
    fn native_single_sgdm_and_adam_follow_independent_nonzero_oracles() {
        let value = [2.0_f32];
        let gradient = [3.0_f32];
        let mut first = [0.0_f32];
        let mut second = [0.0_f32];
        let result = step_values(
            &value,
            &gradient,
            &mut first,
            &mut second,
            &config(Solver::Sgdm),
            1.0,
            0.0,
            1,
            "test",
        )
        .unwrap();
        assert!((result[0] - 1.7).abs() < 1.0e-6);
        assert!((first[0] + 0.3).abs() < 1.0e-6);
        let mut first = [0.0_f32];
        let mut second = [0.0_f32];
        let result = step_values(
            &value,
            &gradient,
            &mut first,
            &mut second,
            &config(Solver::Adam),
            1.0,
            0.0,
            1,
            "test",
        )
        .unwrap();
        assert!((result[0] - 1.9).abs() < 1.0e-6);
        assert!((first[0] - 0.3).abs() < 1.0e-6);
        assert!((second[0] - 0.009).abs() < 1.0e-6);

        let mut first = [0.0_f32];
        let mut second = [];
        let result = step_values(
            &value,
            &gradient,
            &mut first,
            &mut second,
            &config(Solver::Sgdm),
            2.0,
            0.5,
            1,
            "test",
        )
        .unwrap();
        // Effective gradient is 3 + 0.5*2 = 4, effective rate is 0.2.
        assert!((result[0] - 1.2).abs() < 1.0e-6);
    }
}

//! Native-class supervised parameter steps keyed by canonical Learnables roles.
//! Slot state is private to a training invocation; `Layers` remains the only
//! network parameter authority. Neither slots nor network updates are
//! published until every proposed parameter step has succeeded.

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

use super::super::TrainingConfig;
use super::factors::FactorPlan;

mod controls;
mod slot;
mod step;

use controls::NativeControls;
use slot::SolverSlot;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

enum Slot {
    Single(SolverSlot<f32>),
    Double(SolverSlot<f64>),
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

    /// All source identities, shapes, and classes are admitted before any
    /// numerical step. Source tensors are the host snapshots used by forward;
    /// no extra provider download occurs at this boundary.
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
            if shape != gradient.shape() || gradient.is_empty() {
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
                        .class_name()
                        .is(crate::builtins::deep_learning::DLARRAY_CLASS) =>
                {
                    object
                        .properties()
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
            if source.shape() != shape || source.numeric_dtype() != dtype {
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

        // The control plan is per native class and per iteration, never per
        // element. Unused classes do not need materialization.
        let mut single_controls = None;
        let mut double_controls = None;
        let mut staged_slots = Vec::with_capacity(admitted.len());
        let mut updates = Vec::with_capacity(admitted.len());
        for (key, source, gradient, dtype) in admitted {
            let factors = self.factors.get(&key, function)?;
            let gradient = project_owned(gradient, dtype, function)?;
            let (storage, next_slot) = match dtype {
                NumericDType::F32 => {
                    let controls = match single_controls {
                        Some(controls) => controls,
                        None => {
                            let controls =
                                NativeControls::<f32>::admit(config, iteration, function)?;
                            single_controls = Some(controls);
                            controls
                        }
                    };
                    let old = match self.slots.get(&key) {
                        None => None,
                        Some(Slot::Single(slot)) => Some(slot),
                        Some(Slot::Double(_)) => {
                            return Err(fail(function, "optimizer slot precision changed"))
                        }
                    };
                    let native_factors = factors.native::<f32>(
                        config.learn_rate,
                        config.l2_regularization,
                        function,
                    )?;
                    let (values, slot) = step::step(
                        source.as_f32_slice().expect("admitted F32"),
                        gradient.as_f32_slice().expect("projected F32"),
                        old,
                        controls,
                        &native_factors,
                        function,
                    )?;
                    (NumericStorage::F32(values), Slot::Single(slot))
                }
                NumericDType::F64 => {
                    let controls = match double_controls {
                        Some(controls) => controls,
                        None => {
                            let controls =
                                NativeControls::<f64>::admit(config, iteration, function)?;
                            double_controls = Some(controls);
                            controls
                        }
                    };
                    let old = match self.slots.get(&key) {
                        None => None,
                        Some(Slot::Double(slot)) => Some(slot),
                        Some(Slot::Single(_)) => {
                            return Err(fail(function, "optimizer slot precision changed"))
                        }
                    };
                    let native_factors = factors.native::<f64>(
                        config.learn_rate,
                        config.l2_regularization,
                        function,
                    )?;
                    let (values, slot) = step::step(
                        source.as_f64_slice().expect("admitted F64"),
                        gradient.as_f64_slice().expect("projected F64"),
                        old,
                        controls,
                        &native_factors,
                        function,
                    )?;
                    (NumericStorage::F64(values), Slot::Double(slot))
                }
                _ => unreachable!("admitted native precision"),
            };
            let value = Tensor::from_numeric_storage(storage, source.shape().to_vec())
                .map_err(|error| fail(function, error))?;
            staged_slots.push((key.clone(), next_slot));
            updates.push(ParameterUpdate { key, value });
        }
        self.slots
            .try_reserve(staged_slots.len())
            .map_err(|_| fail(function, "cannot reserve optimizer slot storage"))?;
        for (key, slot) in staged_slots {
            self.slots.insert(key, slot);
        }
        Ok(updates)
    }
}

#[cfg(test)]
#[path = "optimizer/tests.rs"]
mod tests;

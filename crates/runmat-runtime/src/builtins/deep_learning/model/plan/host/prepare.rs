//! Selective host materialization after complete descriptor admission.

use std::collections::HashMap;

use runmat_accelerate_api::handle_is_explicit;
use runmat_value::{Tensor, Value};

use super::controls::Controls;
use super::geometry::{HostNetworkPlan, InputPayloadKind};
use crate::builtins::deep_learning::model::{
    execution::host::network::{self, HostExecutionResult, HostStep, HostValue},
    parameters::{ParameterKey, ParameterRole},
    precision::project_owned,
    state::StateRole,
};
use crate::builtins::deep_learning::{deep_learning_error, DLARRAY_CLASS};
use crate::BuiltinResult;

pub(in crate::builtins::deep_learning) struct PreparedHostNetwork<'a> {
    plan: HostNetworkPlan<'a>,
    parameters: HashMap<(usize, ParameterRole), PreparedParameter>,
    state: HashMap<(usize, StateRole), Tensor>,
}

struct PreparedParameter {
    key: ParameterKey,
    execution: Tensor,
    native_source: Option<Tensor>,
}

#[derive(Clone, Copy)]
enum ParameterPreparation {
    ExecutionOnly,
    RetainNativeSource,
}

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

fn data(value: &Value) -> &Value {
    match value {
        Value::Object(object) if object.class_name().is(DLARRAY_CLASS) => object
            .properties()
            .get(runmat_builtins::DLARRAY_DATA_PROPERTY)
            .unwrap_or(value),
        _ => value,
    }
}

async fn host_tensor(value: &Value, function: &'static str) -> BuiltinResult<Tensor> {
    let data = data(value);
    let prepared = match data {
        Value::GpuTensor(handle) => {
            if handle_is_explicit(handle) {
                return Err(fail(
                    function,
                    "explicit gpuArray input cannot be gathered by host network execution",
                ));
            }
            crate::gather_if_needed_async(data).await?
        }
        _ => data.clone(),
    };
    crate::builtins::common::tensor::value_into_tensor_for(function, prepared)
        .map_err(|error| fail(function, error))
}

impl<'a> HostNetworkPlan<'a> {
    /// Input-only preparation does not touch the network's Learnables. A
    /// supervised caller can prepare its full predictor once, then prepare
    /// parameters only for the minibatch that actually executes.
    pub(in crate::builtins::deep_learning) async fn prepare_input(
        &self,
        input: &Value,
    ) -> BuiltinResult<HostValue> {
        prepare_input(self, input).await
    }

    pub(in crate::builtins::deep_learning) async fn prepare(
        self,
    ) -> BuiltinResult<PreparedHostNetwork<'a>> {
        self.prepare_parameters(ParameterPreparation::ExecutionOnly)
            .await
    }

    pub(in crate::builtins::deep_learning) async fn prepare_for_training(
        self,
    ) -> BuiltinResult<PreparedHostNetwork<'a>> {
        self.prepare_parameters(ParameterPreparation::RetainNativeSource)
            .await
    }

    async fn prepare_parameters(
        self,
        retention: ParameterPreparation,
    ) -> BuiltinResult<PreparedHostNetwork<'a>> {
        let mut parameters = HashMap::with_capacity(self.parameters.len());
        for row in &self.parameters {
            let value = row
                .value
                .ok_or_else(|| fail(self.function, "network has an uninitialized parameter"))?;
            let original = host_tensor(value, self.function).await?;
            let native_source = match retention {
                ParameterPreparation::ExecutionOnly => None,
                ParameterPreparation::RetainNativeSource => Some(original.clone()),
            };
            let execution = project_owned(original, self.precision, self.function)?;
            parameters.insert(
                (row.key.layer_index, row.key.role),
                PreparedParameter {
                    key: row.key.clone(),
                    execution,
                    native_source,
                },
            );
        }
        let mut state = HashMap::with_capacity(self.state.len());
        for row in &self.state {
            let prepared = project_owned(
                host_tensor(row.value, self.function).await?,
                self.precision,
                self.function,
            )?;
            state.insert((row.key.layer_index, row.key.role), prepared);
        }
        Ok(PreparedHostNetwork {
            plan: self,
            parameters,
            state,
        })
    }
}

impl PreparedHostNetwork<'_> {
    pub(in crate::builtins::deep_learning) async fn prepare_input(
        &self,
        input: &Value,
    ) -> BuiltinResult<HostValue> {
        prepare_input(&self.plan, input).await
    }

    /// Borrowed execution snapshot, keyed by canonical roles. Cloning a
    /// Tensor here pins its Arc/CoW payload; it does not create a persistent
    /// second parameter store or redownload a resident value.
    pub(in crate::builtins::deep_learning) fn original_parameters(
        &self,
    ) -> Vec<(ParameterKey, Tensor)> {
        self.plan
            .parameters
            .iter()
            .map(|row| {
                let parameter = self
                    .parameters
                    .get(&(row.key.layer_index, row.key.role))
                    .expect("prepared every admitted parameter");
                let tensor = parameter
                    .native_source
                    .as_ref()
                    .expect("training preparation retains native source")
                    .clone();
                (parameter.key.clone(), tensor)
            })
            .collect()
    }

    fn parameter(&self, index: usize, role: ParameterRole) -> &Tensor {
        self.parameters
            .get(&(index, role))
            .map(|parameter| &parameter.execution)
            .expect("pure plan admitted every consumed parameter")
    }

    fn steps(&self) -> Vec<HostStep<'_>> {
        self.plan
            .steps
            .iter()
            .map(|descriptor| {
                let index = descriptor.index;
                match descriptor.controls {
                    Controls::Input { .. } => HostStep::Input,
                    Controls::Dense => HostStep::FullyConnected {
                        weights: self.parameter(index, ParameterRole::Weights),
                        bias: self.parameter(index, ParameterRole::Bias),
                    },
                    Controls::Convolution(options) => HostStep::Convolution1d {
                        weights: self.parameter(index, ParameterRole::Weights),
                        bias: self.parameter(index, ParameterRole::Bias),
                        options,
                    },
                    Controls::Recurrent(options) => HostStep::Recurrent {
                        input_weights: self.parameter(index, ParameterRole::InputWeights),
                        recurrent_weights: self.parameter(index, ParameterRole::RecurrentWeights),
                        bias: self.parameter(index, ParameterRole::Bias),
                        initial_hidden: self.state.get(&(index, StateRole::HiddenState)),
                        initial_cell: self.state.get(&(index, StateRole::CellState)),
                        options,
                    },
                    Controls::Normalization(options) => HostStep::LayerNormalization {
                        scale: self.parameter(index, ParameterRole::Scale),
                        offset: self.parameter(index, ParameterRole::Offset),
                        options,
                    },
                    Controls::Pooling => HostStep::GlobalAveragePooling1d,
                    Controls::Relu => HostStep::Relu,
                    Controls::Elu { alpha } => HostStep::Elu { alpha },
                    Controls::Softmax => HostStep::Softmax,
                    Controls::Output => match descriptor.kind {
                        runmat_builtins::DeepLearningLayerKind::ClassificationOutput => {
                            HostStep::ClassificationOutput
                        }
                        _ => HostStep::RegressionOutput,
                    },
                }
            })
            .collect()
    }

    pub(in crate::builtins::deep_learning) fn execute(
        &self,
        input: HostValue,
        capture: bool,
    ) -> BuiltinResult<HostExecutionResult> {
        let result = network::execute(&self.steps(), input, self.plan.input_axes.clone(), capture)
            .map_err(|error| fail(self.plan.function, error))?;
        if result.axes != self.plan.output_axes {
            return Err(fail(
                self.plan.function,
                "executed output geometry differs from admitted plan",
            ));
        }
        Ok(result)
    }
}

async fn prepare_input(plan: &HostNetworkPlan<'_>, input: &Value) -> BuiltinResult<HostValue> {
    if plan.input_payload_kind == InputPayloadKind::Complex {
        let Value::ComplexTensor(tensor) = data(input) else {
            return Err(fail(
                plan.function,
                "prepared input differs from admitted complex payload",
            ));
        };
        if tensor.shape() != plan.input_axes.shape()
            || tensor.numeric_dtype() != plan.precision
            || tensor.len() != plan.input_axes.element_count()
        {
            return Err(fail(
                plan.function,
                "prepared complex input differs from admitted shape or precision",
            ));
        }
        // ComplexTensor owns an Arc/CoW host allocation. This pins the
        // admitted class and bytes without a host copy or demotion.
        return Ok(HostValue::Complex(tensor.clone()));
    }
    let tensor = project_owned(
        host_tensor(input, plan.function).await?,
        plan.precision,
        plan.function,
    )?;
    if tensor.shape() != plan.input_axes.shape() {
        return Err(fail(
            plan.function,
            "prepared input shape differs from admitted input",
        ));
    }
    Ok(HostValue::Real(tensor))
}

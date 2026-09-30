//! Selective host materialization after complete descriptor admission.

use std::collections::HashMap;

use runmat_accelerate_api::handle_is_explicit;
use runmat_value::{NumericDType, NumericStorage, Tensor, Value};

use super::controls::Controls;
use super::geometry::HostNetworkPlan;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;
use crate::builtins::deep_learning::model::{
    execution::host::network::{self, HostExecutionResult, HostStep, HostValue},
    parameters::{ParameterRole, ParameterView},
    state::{StateRole, StateView},
};
use crate::builtins::deep_learning::{deep_learning_error, DLARRAY_CLASS};
use crate::BuiltinResult;

pub(in crate::builtins::deep_learning) struct PreparedHostNetwork<'a> {
    plan: HostNetworkPlan<'a>,
    parameters: HashMap<(usize, ParameterRole), Tensor>,
    state: HashMap<(usize, StateRole), Tensor>,
}

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

fn data(value: &Value) -> &Value {
    match value {
        Value::Object(object) if object.class_name.is(DLARRAY_CLASS) => {
            object.properties.get("Data").unwrap_or(value)
        }
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

fn project(tensor: Tensor, dtype: NumericDType, function: &'static str) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == dtype {
        return Ok(tensor);
    }
    let shape = tensor.shape.clone();
    let storage = match dtype {
        NumericDType::F32 => NumericStorage::F32(
            tensor
                .into_numeric_storage()
                .map_err(|error| fail(function, error))?
                .materialize_f32(),
        ),
        NumericDType::F64 => NumericStorage::F64(tensor.materialize_f64()),
        _ => {
            return Err(fail(
                function,
                "host execution requires single or double precision",
            ))
        }
    };
    Tensor::from_numeric_storage(storage, shape).map_err(|error| fail(function, error))
}

impl<'a> HostNetworkPlan<'a> {
    pub(in crate::builtins::deep_learning) async fn prepare(
        self,
    ) -> BuiltinResult<PreparedHostNetwork<'a>> {
        let mut parameters = HashMap::with_capacity(self.parameters.len());
        for row in &self.parameters {
            let value = row
                .value
                .ok_or_else(|| fail(self.function, "network has an uninitialized parameter"))?;
            let prepared = project(
                host_tensor(value, self.function).await?,
                self.precision,
                self.function,
            )?;
            parameters.insert((row.key.layer_index, row.key.role), prepared);
        }
        let mut state = HashMap::with_capacity(self.state.len());
        for row in &self.state {
            let prepared = project(
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
    pub(in crate::builtins::deep_learning) fn parameter_rows(&self) -> &[ParameterView<'_>] {
        &self.plan.parameters
    }

    pub(in crate::builtins::deep_learning) fn state_rows(&self) -> &[StateView<'_>] {
        &self.plan.state
    }

    pub(in crate::builtins::deep_learning) fn output_axes(&self) -> &AxisPlan {
        &self.plan.output_axes
    }

    pub(in crate::builtins::deep_learning) async fn prepare_input(
        &self,
        input: &Value,
    ) -> BuiltinResult<HostValue> {
        let tensor = project(
            host_tensor(input, self.plan.function).await?,
            self.plan.precision,
            self.plan.function,
        )?;
        if tensor.shape != self.plan.input_axes.shape() {
            return Err(fail(
                self.plan.function,
                "prepared input shape differs from admitted input",
            ));
        }
        Ok(HostValue::Real(tensor))
    }

    fn parameter(&self, index: usize, role: ParameterRole) -> &Tensor {
        self.parameters
            .get(&(index, role))
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

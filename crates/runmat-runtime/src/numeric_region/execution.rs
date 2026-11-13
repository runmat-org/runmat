//! One numeric evaluator with policy-owned physical outputs. Bounded DAG
//! scratch and output-owner slots live on the stack; managed payloads, shape
//! owners and the final collection are admitted before allocation.
use super::*;
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::execution::RootedValues;
use crate::{runtime_error::semantic_error, RuntimeError};
use runmat_value::{ExactHostTensorAllocation, NumericDType, OutputListStorage};

#[derive(Debug)]
pub enum NumericRegionOwnedExecution {
    Completed(RootedValues),
    Ineligible,
    Cancelled,
}

#[cfg(test)]
mod tests;

pub(super) enum StorageExecution {
    Completed(OutputListStorage),
    Ineligible,
    Cancelled,
}

pub fn execute_in_context(
    program: &NumericRegionProgram,
    inputs: &[&Value],
    runtime: &RuntimeContext,
) -> Result<NumericRegionOwnedExecution, RuntimeError> {
    match run(program, inputs, &runtime.cancellation(), Some(runtime))? {
        StorageExecution::Ineligible => Ok(NumericRegionOwnedExecution::Ineligible),
        StorageExecution::Cancelled => Ok(NumericRegionOwnedExecution::Cancelled),
        StorageExecution::Completed(values) => {
            RootedValues::from_output_storage_in_context(values, runtime)
                .map(NumericRegionOwnedExecution::Completed)
        }
    }
}

/// Same final-output bound supplied to the existing native candidate graph.
/// Stack arrays need no heap scratch; generic native execution keeps its own
/// explicitly unknown intermediate/scratch dimension.
pub fn output_retained_bytes(
    workload: NumericRegionWorkload,
    outputs: usize,
) -> Result<u64, HostAllocationError> {
    let one = workload
        .output_rank
        .map(|rank| {
            HostAllocationPlan::retained_tensor_bytes_for_dimensions(
                NumericDType::F64,
                workload.elements,
                rank,
            )
        })
        .transpose()?
        .unwrap_or(0);
    let tensors = one
        .checked_mul(u64::try_from(outputs).map_err(|_| HostAllocationError::Overflow)?)
        .ok_or(HostAllocationError::Overflow)?;
    let slots = OutputListStorage::requested_bytes(outputs)?;
    let slots = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(slots).map_err(|_| HostAllocationError::Overflow)?,
        usize::from(outputs != 0),
    )?;
    tensors
        .checked_add(slots)
        .ok_or(HostAllocationError::Overflow)
}

enum OutputBuffer {
    Exact(ExactHostTensorAllocation),
    Ordinary(Vec<f64>),
    Scalar(f64),
}

/// Source storage of the two bounded evaluator arrays. Limits are the
/// existing validator's 256 nodes / 64 outputs, not new input thresholds.
pub const NUMERIC_REGION_EVALUATOR_ARRAY_BYTES: usize = std::mem::size_of::<[f64; MAX_NODES]>()
    + std::mem::size_of::<[Option<OutputBuffer>; MAX_OUTPUTS]>();
const _: () = assert!(NUMERIC_REGION_EVALUATOR_ARRAY_BYTES <= 16_384);

impl OutputBuffer {
    fn push(&mut self, index: usize, value: f64) {
        match self {
            Self::Exact(output) => {
                output.as_f64_slice_mut().expect("F64 allocation")[index] = value
            }
            Self::Ordinary(output) => output.push(value),
            Self::Scalar(output) => *output = value,
        }
    }

    fn finish(self, shape: Option<&[usize]>) -> Result<Value, HostAllocationError> {
        match self {
            Self::Exact(output) => output.into_tensor().map(Value::Tensor).map_err(Into::into),
            Self::Ordinary(output) => {
                Tensor::new(output, shape.expect("dense output shape").to_vec())
                    .map(Value::Tensor)
                    .map_err(|_| HostAllocationError::Overflow)
            }
            Self::Scalar(output) => Ok(Value::Num(output)),
        }
    }
}

fn allocation_error(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:Native:NumericRegion",
        "numeric region physical output admission failed",
        source,
    )
}

fn allocation_failure(
    runtime: Option<&RuntimeContext>,
    receipt: Option<&HostAllocationReceipt>,
    source: HostAllocationError,
) -> RuntimeError {
    if let (Some(runtime), Some(receipt)) = (runtime, receipt) {
        if let Err(cause) = receipt.checkpoint(runtime) {
            return allocation_error(Some(runtime), cause);
        }
    }
    allocation_error(runtime, source)
}

pub(super) fn run(
    program: &NumericRegionProgram,
    inputs: &[&Value],
    cancellation: &CancellationToken,
    runtime: Option<&RuntimeContext>,
) -> Result<StorageExecution, RuntimeError> {
    if cancellation.is_cancelled() {
        return Ok(StorageExecution::Cancelled);
    }
    program
        .validate(inputs.len())
        .map_err(|message| semantic_error("RunMat:Native:NumericRegion", message))?;
    let Some(shape) = analyze_inputs(inputs) else {
        return Ok(StorageExecution::Ineligible);
    };
    let workload =
        workload(inputs).ok_or_else(|| allocation_error(runtime, HostAllocationError::Overflow))?;
    let retained = output_retained_bytes(workload, program.outputs.len())
        .map_err(|source| allocation_error(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, retained).map_err(|source| allocation_error(runtime, source))?,
    )
    .map_err(|source| allocation_error(runtime, source))?;
    let permit = receipt
        .as_ref()
        .map(|receipt| receipt.output_permit(runtime.expect("managed runtime")))
        .transpose()
        .map_err(|source| allocation_error(runtime, source))?;
    let mut outputs: [Option<OutputBuffer>; MAX_OUTPUTS] = [const { None }; MAX_OUTPUTS];
    for output in &mut outputs[..program.outputs.len()] {
        *output = Some(match (shape, permit.as_ref()) {
            (Some(shape), Some(permit)) => OutputBuffer::Exact(
                ExactHostTensorAllocation::new(NumericDType::F64, shape, permit).map_err(
                    |source| allocation_failure(runtime, receipt.as_ref(), source.into()),
                )?,
            ),
            (Some(_), None) => OutputBuffer::Ordinary(Vec::with_capacity(workload.elements)),
            (None, _) => OutputBuffer::Scalar(0.0),
        });
    }
    let mut values = [0.0; MAX_NODES];
    for element in 0..workload.elements {
        if element % 1_024 == 0 {
            if cancellation.is_cancelled() {
                return Ok(StorageExecution::Cancelled);
            }
            if let Some(receipt) = &receipt {
                receipt
                    .checkpoint(runtime.expect("managed runtime"))
                    .map_err(|source| allocation_error(runtime, source))?;
            }
        }
        for (index, node) in program.nodes.iter().enumerate() {
            values[index] = match *node {
                NumericRegionNode::Input(input) => match inputs[input] {
                    Value::Num(value) => *value,
                    Value::Tensor(tensor) => {
                        let data = tensor.as_f64_slice().expect("validated F64 input");
                        data[if tensor.len() == 1 { 0 } else { element }]
                    }
                    _ => unreachable!("validated numeric input"),
                },
                NumericRegionNode::Constant(value) => value,
                NumericRegionNode::Unary { operation, input } => match operation {
                    NumericUnaryOperation::Plus => values[input],
                    NumericUnaryOperation::Minus => -values[input],
                },
                NumericRegionNode::Binary {
                    operation,
                    left,
                    right,
                } => match operation {
                    NumericBinaryOperation::Add => values[left] + values[right],
                    NumericBinaryOperation::Subtract => values[left] - values[right],
                    NumericBinaryOperation::Multiply => values[left] * values[right],
                    NumericBinaryOperation::Divide => values[left] / values[right],
                    NumericBinaryOperation::LeftDivide => values[right] / values[left],
                },
            };
        }
        for (output, node) in outputs.iter_mut().zip(&program.outputs) {
            output
                .as_mut()
                .expect("initialized output")
                .push(element, values[*node]);
        }
    }
    if cancellation.is_cancelled() {
        return Ok(StorageExecution::Cancelled);
    }
    let mut finish = |index: usize| {
        outputs[index]
            .take()
            .expect("single output adoption")
            .finish(shape)
            .map_err(|source| allocation_failure(runtime, receipt.as_ref(), source))
    };
    let values = if let Some(permit) = &permit {
        OutputListStorage::try_initialize(program.outputs.len(), permit, &mut finish).map_err(
            |source| match source {
                runmat_value::OutputListInitializationError::Allocation(source) => {
                    allocation_failure(runtime, receipt.as_ref(), source.into())
                }
                runmat_value::OutputListInitializationError::Element(source) => source,
            },
        )?
    } else {
        OutputListStorage::unbounded(
            (0..program.outputs.len())
                .map(finish)
                .collect::<Result<Vec<_>, _>>()?,
        )
    };
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(runtime.expect("managed runtime"))
            .map_err(|source| allocation_error(runtime, source))?;
    }
    Ok(StorageExecution::Completed(values))
}

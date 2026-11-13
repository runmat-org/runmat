//! The already-verified native numeric subset joins the existing typed fusion
//! producer and selected-provider execution owner. This is not a second graph
//! planner, a bytecode installation, or a native-to-Accelerate dependency.
use crate::fusion::{FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp};
use crate::fusion_exec::FusionExecutionRequest;
use crate::graph::{InstrSpan, PrimitiveOp, ShapeInfo};
use runmat_runtime::context::{
    host_allocation::{HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_runtime::numeric_region::service::{
    NativeNumericRegionOutcome, NativeNumericRegionRequest,
};
use runmat_runtime::numeric_region::{
    NumericBinaryOperation, NumericRegionNode, NumericUnaryOperation,
};
use runmat_runtime::{runtime_error::semantic_error, RuntimeError};
use runmat_value::Value;
use std::collections::HashMap;

fn error(message: impl Into<String>) -> RuntimeError {
    semantic_error("RunMat:Native:NumericRegion", message)
}

fn allocation_error(
    runtime: &RuntimeContext,
    source: runmat_runtime::context::host_allocation::HostAllocationError,
) -> RuntimeError {
    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
        Some(runtime),
        "RunMat:Native:NumericRegion",
        "native fusion metadata admission failed",
        source,
    )
}

pub(super) async fn execute(
    runtime: RuntimeContext,
    request: NativeNumericRegionRequest,
) -> Result<NativeNumericRegionOutcome, RuntimeError> {
    if runtime
        .acceleration_policy()
        .is_some_and(|policy| !policy.automatic_offload_enabled())
    {
        return Ok(NativeNumericRegionOutcome::Ineligible);
    }
    let (program_owner, inputs, right) = request.into_parts();
    program_owner
        .checkpoint(&runtime)
        .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
        .map_err(|source| {
            runmat_runtime::runtime_error::with_provider_execution_source(
                error("native program allocation authority changed"),
                source,
            )
        })?;
    let program = program_owner.program();
    if !right.is_current(&runtime) {
        return Err(error("native numeric region execution generation changed"));
    }
    program.validate(inputs.len()).map_err(error)?;
    // Reserve the new plan/input/output slot metadata before allocating it.
    // Constants become owned scalar inputs; no HashMap or constant Value
    // forest is manufactured, and all tensor clones retain shared owners.
    let n = program.nodes.len();
    let bytes_per_node = std::mem::size_of::<FusionOp>()
        .checked_add(std::mem::size_of::<Value>())
        .and_then(|bytes| bytes.checked_add(4 * std::mem::size_of::<u32>()))
        .ok_or_else(|| error("native fusion metadata quote overflow"))?;
    let bytes = n
        .checked_mul(bytes_per_node)
        .and_then(|bytes| {
            program
                .outputs
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|outputs| bytes.checked_add(outputs))
        })
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| error("native fusion metadata quote overflow"))?;
    let metadata = HostAllocationReceipt::admit(
        Some(&runtime),
        HostAllocationPlan::new(bytes, 0).map_err(|source| allocation_error(&runtime, source))?,
    )
    .map_err(|source| allocation_error(&runtime, source))?;
    let mut operations = Vec::with_capacity(n);
    let mut execution_inputs = Vec::with_capacity(n);
    let mut input_ids = Vec::with_capacity(n);
    let mut nodes = Vec::with_capacity(n);
    for (index, node) in program.nodes.iter().enumerate() {
        let id =
            u32::try_from(index).map_err(|_| error("native fusion value identity overflow"))?;
        let mut retain_input = |value: Value| {
            input_ids.push(id);
            execution_inputs.push(value);
        };
        let (op, operands) = match node {
            NumericRegionNode::Input(input) => {
                match inputs.values().get(*input) {
                    Some(Value::Num(value)) => retain_input(Value::Num(*value)),
                    Some(Value::Tensor(tensor))
                        if matches!(
                            tensor.numeric_dtype(),
                            runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
                        ) =>
                    {
                        retain_input(Value::Tensor(tensor.clone()))
                    }
                    Some(Value::GpuTensor(handle)) => {
                        retain_input(Value::GpuTensor(handle.clone()))
                    }
                    _ => {
                        return Err(error(
                            "native numeric region input escaped its closed representation",
                        ))
                    }
                }
                continue;
            }
            NumericRegionNode::Constant(value) => {
                retain_input(Value::Num(*value));
                continue;
            }
            NumericRegionNode::Unary { operation, input } => (
                match operation {
                    NumericUnaryOperation::Plus => PrimitiveOp::UPlus,
                    NumericUnaryOperation::Minus => PrimitiveOp::Neg,
                },
                vec![u32::try_from(*input)
                    .map_err(|_| error("native fusion operand identity overflow"))?],
            ),
            NumericRegionNode::Binary {
                operation,
                left,
                right,
            } => (
                match operation {
                    NumericBinaryOperation::Add => PrimitiveOp::Add,
                    NumericBinaryOperation::Subtract => PrimitiveOp::Sub,
                    NumericBinaryOperation::Multiply => PrimitiveOp::ElemMul,
                    NumericBinaryOperation::Divide => PrimitiveOp::ElemDiv,
                    NumericBinaryOperation::LeftDivide => PrimitiveOp::ElemLeftDiv,
                },
                vec![
                    u32::try_from(*left)
                        .map_err(|_| error("native fusion operand identity overflow"))?,
                    u32::try_from(*right)
                        .map_err(|_| error("native fusion operand identity overflow"))?,
                ],
            ),
        };
        operations.push(FusionOp::Primitive {
            op,
            inputs: operands,
            output: Some(id),
        });
        nodes.push(id);
    }
    let output_ids = program
        .outputs
        .iter()
        .map(|output| {
            u32::try_from(*output).map_err(|_| error("native fusion output identity overflow"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let plan = FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes,
            shape: ShapeInfo::Unknown,
            span: InstrSpan { start: 0, end: 0 },
            pattern: None,
            stack_layout: None,
        },
        operations,
        inputs: input_ids,
        stack_pattern: Vec::new(),
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: Vec::new(),
        output: output_ids.first().copied(),
        kernel: FusionKernelSpec {
            kind: FusionKind::ElementwiseChain,
            supported: true,
        },
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    };
    if let Some(metadata) = &metadata {
        metadata
            .checkpoint(&runtime)
            .map_err(|source| allocation_error(&runtime, source))?;
    }
    let outcome = crate::fusion_exec::execute_elementwise_native(
        FusionExecutionRequest {
            plan: &plan,
            inputs: execution_inputs,
            placement: None,
            runtime: Some(runtime),
        },
        &output_ids,
        right,
        &program_owner,
    )
    .await
    .map_err(|source| {
        runmat_runtime::runtime_error::with_provider_execution_source(
            error("native numeric region provider execution failed"),
            source,
        )
    })?;
    // Plan/request slot backing retires before the matching scratch receipt.
    drop(plan);
    drop(output_ids);
    drop(inputs);
    drop(metadata);
    drop(program_owner);
    Ok(outcome)
}

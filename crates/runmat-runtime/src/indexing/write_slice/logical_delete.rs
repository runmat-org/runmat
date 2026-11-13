//! Logical deletion keeps logical bytes native and admits each physical owner.

use super::{
    canonicalize_delete_positions_desc, deleted_vector_dimensions, dense_delete_dimensions,
    is_empty_delete_rhs, validated_delete_position, IndexPlan,
};
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::{ExactHostLogicalTensorAllocation, LogicalArray, Value};

const CHUNK: usize = 4096;

enum LogicalInput {
    Array(LogicalArray),
    Scalar(u8),
}

impl LogicalInput {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Array(value) => &value.shape,
            Self::Scalar(_) => &[1, 1],
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Array(value) => &value.data,
            Self::Scalar(value) => std::slice::from_ref(value),
        }
    }
}

struct Admission<'a> {
    receipt: HostAllocationReceipt,
    runtime: &'a RuntimeContext,
}

enum PositionStorage {
    Empty,
    Ordinary(Vec<usize>),
    Managed(HostTypedWorkspace<usize>),
}

struct Positions<'a> {
    // Physical storage retires before its external admission owner.
    storage: PositionStorage,
    admission: Option<Admission<'a>>,
}

impl<'a> Positions<'a> {
    fn empty() -> Self {
        Self {
            storage: PositionStorage::Empty,
            admission: None,
        }
    }

    fn allocate(runtime: Option<&'a RuntimeContext>, len: usize) -> Result<Self, RuntimeError> {
        if len == 0 {
            return Ok(Self::empty());
        }
        let scratch = HostTypedWorkspace::<usize>::quoted_bytes(len)
            .map_err(|cause| allocation(runtime, cause))?;
        let plan =
            HostAllocationPlan::new(scratch, 0).map_err(|cause| allocation(runtime, cause))?;
        match HostAllocationReceipt::admit(runtime, plan)
            .map_err(|cause| allocation(runtime, cause))?
        {
            Some(receipt) => {
                let runtime = runtime
                    .ok_or_else(|| allocation(runtime, HostAllocationError::MissingManagedOwner))?;
                let values = receipt
                    .temporary_typed(runtime, len)
                    .map_err(|cause| allocation(Some(runtime), cause))?;
                Ok(Self {
                    storage: PositionStorage::Managed(values),
                    admission: Some(Admission { receipt, runtime }),
                })
            }
            None => Ok(Self {
                storage: PositionStorage::Ordinary(vec![0; len]),
                admission: None,
            }),
        }
    }

    fn as_slice(&self) -> &[usize] {
        match &self.storage {
            PositionStorage::Empty => &[],
            PositionStorage::Ordinary(values) => values,
            PositionStorage::Managed(values) => values.as_slice(),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [usize] {
        match &mut self.storage {
            PositionStorage::Empty => &mut [],
            PositionStorage::Ordinary(values) => values,
            PositionStorage::Managed(values) => values.as_mut_slice(),
        }
    }

    fn checkpoint(&self) -> Result<(), RuntimeError> {
        if let Some(Admission { receipt, runtime }) = &self.admission {
            receipt
                .checkpoint(runtime)
                .map_err(|cause| allocation(Some(runtime), cause))?;
        }
        Ok(())
    }
}

enum LogicalStorage {
    Ordinary(Vec<u8>),
    Managed(ExactHostLogicalTensorAllocation),
}

struct LogicalOutput<'a> {
    // Physical storage retires before its external admission owner.
    storage: LogicalStorage,
    admission: Option<Admission<'a>>,
}

impl<'a> LogicalOutput<'a> {
    fn allocate(
        runtime: Option<&'a RuntimeContext>,
        shape: &[usize],
    ) -> Result<Self, RuntimeError> {
        let retained = HostAllocationPlan::retained_logical_tensor_bytes(shape)
            .map_err(|cause| allocation(runtime, cause))?;
        let plan =
            HostAllocationPlan::new(0, retained).map_err(|cause| allocation(runtime, cause))?;
        match HostAllocationReceipt::admit(runtime, plan)
            .map_err(|cause| allocation(runtime, cause))?
        {
            Some(receipt) => {
                let runtime = runtime
                    .ok_or_else(|| allocation(runtime, HostAllocationError::MissingManagedOwner))?;
                let values = receipt
                    .with_output_allocation(runtime, |permit| {
                        ExactHostLogicalTensorAllocation::new(shape, permit)
                    })
                    .map_err(|cause| allocation(Some(runtime), cause))?;
                Ok(Self {
                    storage: LogicalStorage::Managed(values),
                    admission: Some(Admission { receipt, runtime }),
                })
            }
            None => Ok(Self {
                storage: LogicalStorage::Ordinary(vec![0; shape.iter().product()]),
                admission: None,
            }),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match &mut self.storage {
            LogicalStorage::Ordinary(values) => values,
            LogicalStorage::Managed(values) => values.as_mut_slice(),
        }
    }

    fn admission(&self) -> Option<(&HostAllocationReceipt, &RuntimeContext)> {
        self.admission
            .as_ref()
            .map(|admission| (&admission.receipt, admission.runtime))
    }

    async fn into_array(
        self,
        shape: &[usize],
        runtime: Option<&RuntimeContext>,
        budget: &mut Option<CpuCooperationBudget>,
        positions: &Positions<'_>,
    ) -> Result<LogicalArray, RuntimeError> {
        let Self { storage, admission } = self;
        match storage {
            LogicalStorage::Ordinary(values) => {
                let array = LogicalArray::new(values, shape.to_vec())
                    .map_err(|detail| semantic_error("ShapeMismatch", detail))?;
                checkpoint(runtime, budget, positions, None, 0).await?;
                Ok(array)
            }
            LogicalStorage::Managed(values) => {
                let Admission { receipt, runtime } =
                    admission.expect("managed logical output retains its admission");
                let array = receipt
                    .with_output_allocation(runtime, |_| values.into_array())
                    .map_err(|cause| allocation(Some(runtime), cause))?;
                checkpoint(
                    Some(runtime),
                    budget,
                    positions,
                    Some((&receipt, runtime)),
                    0,
                )
                .await?;
                Ok(array)
            }
        }
    }
}

fn allocation(runtime: Option<&RuntimeContext>, cause: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:Indexing:LogicalDeletion",
        "logical deletion admission failed",
        cause,
    )
}

async fn checkpoint(
    runtime: Option<&RuntimeContext>,
    budget: &mut Option<CpuCooperationBudget>,
    positions: &Positions<'_>,
    output: Option<(&HostAllocationReceipt, &RuntimeContext)>,
    work: usize,
) -> Result<(), RuntimeError> {
    positions.checkpoint()?;
    if let Some((receipt, runtime)) = output {
        receipt
            .checkpoint(runtime)
            .map_err(|cause| allocation(Some(runtime), cause))?;
    }
    if let Some(budget) = budget {
        let work =
            u64::try_from(work).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?;
        budget
            .checkpoint(work)
            .await
            .map_err(|cause| cause.into_runtime_error_in_context(runtime))?;
    }
    positions.checkpoint()?;
    if let Some((receipt, runtime)) = output {
        receipt
            .checkpoint(runtime)
            .map_err(|cause| allocation(Some(runtime), cause))?;
    }
    Ok(())
}

pub async fn delete_logical_with_plan(
    context: Option<&RuntimeContext>,
    value: LogicalArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    delete(context, LogicalInput::Array(value), plan, rhs).await
}

/// A logical scalar enters without first allocating an unadmitted array.
pub async fn delete_bool_with_plan(
    context: Option<&RuntimeContext>,
    value: bool,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    delete(context, LogicalInput::Scalar(u8::from(value)), plan, rhs).await
}

async fn delete(
    context: Option<&RuntimeContext>,
    value: LogicalInput,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    // Preserve the public ambient compatibility path without overriding an
    // explicitly selected invocation with another ambient context.
    let selected = context.cloned().or_else(crate::context::legacy::active);
    let runtime = selected.as_ref();
    let owner = HostAllocationReceipt::capture_owner(runtime)
        .map_err(|cause| allocation(runtime, cause))?;
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    checkpoint(runtime, &mut budget, &Positions::empty(), None, 0).await?;
    if !is_empty_delete_rhs(rhs) {
        return Err(semantic_error(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    if plan.indices.is_empty() {
        if let (Some(owner), Some(runtime)) = (&owner, runtime) {
            owner
                .check(runtime)
                .map_err(|cause| allocation(Some(runtime), cause))?;
        }
        return Ok(match value {
            LogicalInput::Array(value) => Value::LogicalArray(value),
            LogicalInput::Scalar(value) => Value::Bool(value != 0),
        });
    }
    let (rows, cols) = dense_delete_dimensions(value.shape());
    if !(rows == 1 || cols == 1) {
        return Err(semantic_error(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    let mut positions = Positions::allocate(runtime, plan.indices.len())?;
    for (chunk, indices) in plan.indices.chunks(CHUNK).enumerate() {
        let start = chunk * CHUNK;
        for (&index, target) in indices
            .iter()
            .zip(&mut positions.as_mut_slice()[start..start + indices.len()])
        {
            *target = validated_delete_position(index, value.bytes().len())?;
        }
        checkpoint(runtime, &mut budget, &positions, None, indices.len()).await?;
    }
    positions.checkpoint()?;
    // Same existing indivisible in-place sort as numeric deletion. Check
    // cancellation/owner immediately around it, not per-comparison claims.
    let unique = canonicalize_delete_positions_desc(positions.as_mut_slice());
    checkpoint(runtime, &mut budget, &positions, None, plan.indices.len()).await?;
    let len = value
        .bytes()
        .len()
        .checked_sub(unique)
        .ok_or_else(|| semantic_error("IndexOutOfBounds", "Index out of bounds"))?;
    let shape = deleted_vector_dimensions(rows, len);
    let mut output = LogicalOutput::allocate(runtime, &shape)?;
    let mut removed = positions.as_slice()[..unique]
        .iter()
        .rev()
        .copied()
        .peekable();
    let mut destination = 0;
    for (index, &byte) in value.bytes().iter().enumerate() {
        if removed.peek() == Some(&index) {
            removed.next();
        } else {
            output.as_mut_slice()[destination] = u8::from(byte != 0);
            destination += 1;
        }
        if (index + 1).is_multiple_of(CHUNK) {
            checkpoint(runtime, &mut budget, &positions, output.admission(), CHUNK).await?;
        }
    }
    checkpoint(
        runtime,
        &mut budget,
        &positions,
        output.admission(),
        value.bytes().len() % CHUNK,
    )
    .await?;
    let output = output
        .into_array(&shape, runtime, &mut budget, &positions)
        .await?;
    if let (Some(owner), Some(runtime)) = (&owner, runtime) {
        owner
            .check(runtime)
            .map_err(|cause| allocation(Some(runtime), cause))?;
    }
    Ok(Value::LogicalArray(output))
}

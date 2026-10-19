use runmat_mir::{MirIndexComponent, MirIndexing, MirOperand};
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
};
use runmat_runtime::execution::{RootedValueLease, RootedValues};
use runmat_runtime::indexing::plan::IndexPlan;
use runmat_runtime::indexing::read_slice;
use runmat_runtime::object::dispatch::resolve_object_index_protocol;
use runmat_runtime::object::indexing::{
    ObjectIndexComponent, ObjectIndexOp, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use runmat_runtime::RuntimeError;
use runmat_types::{IndexKind, IndexResultContext};
use runmat_value::{LogicalArray, SymbolicArray, Value};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::{materialize_operand, materialize_retained_operand};
use super::state::HostState;

mod assignment_operation;
mod read_operation;

/// A direct host completion allocates selector and result backing on the
/// generated-code stack. Managed execution must use the admitted resumable
/// operation owner instead; this policy probe grants no allocation itself.
fn direct_host_indexing_allowed(state: &HostState) -> NativeExecutorResult<bool> {
    HostAllocationReceipt::policy(Some(&state.runtime))
        .map(|policy| policy == HostAllocationPolicy::Unbounded)
        .map_err(|source| {
            NativeExecutorError::from(host_allocation_runtime_error(
                Some(&state.runtime),
                "RunMat:Indexing:HostAllocation",
                "host indexing policy inspection failed",
                source,
            ))
        })
}

enum IndexedReadOutcome {
    Ordinary(runmat_value::OutputListStorage),
    Host(runmat_runtime::execution::RootedValues),
}

impl IndexedReadOutcome {
    fn values(&self) -> &[Value] {
        match self {
            Self::Ordinary(values) => values,
            Self::Host(values) => values.values(),
        }
    }

    fn into_storage(self) -> runmat_value::OutputListStorage {
        match self {
            Self::Ordinary(values) => values,
            Self::Host(values) => values.with_owned_output_storage(|values| values),
        }
    }
}

enum IndexedAssignmentOutcome {
    Ordinary(Value),
    Host(runmat_runtime::execution::RootedValues),
}

impl IndexedAssignmentOutcome {
    fn values(&self) -> &[Value] {
        match self {
            Self::Ordinary(value) => std::slice::from_ref(value),
            Self::Host(values) => values.values(),
        }
    }

    fn into_value(self) -> NativeExecutorResult<Value> {
        match self {
            Self::Ordinary(value) => Ok(value),
            Self::Host(values) => values.with_owned_output_storage(|values| {
                if values.as_slice().len() != 1 {
                    return Err(NativeExecutorError::Host(
                        "host indexed assignment did not produce one receiver".into(),
                    ));
                }
                super::call_suspension::own_output_storage(values)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        NativeExecutorError::Host(
                            "host indexed assignment lost its receiver".into(),
                        )
                    })
            }),
        }
    }
}

fn object_selector(selectors: &MaterializedSelectors) -> NativeExecutorResult<ObjectIndexSelector> {
    if selectors.end_mask != 0 {
        return Err(NativeExecutorError::from(semantic_error(
            "ContextualEndRequiresSubscriptPath",
            "object end must be evaluated through the contextual subscript path",
        )));
    }
    let mut values = selectors.numeric.iter();
    let mut components = Vec::with_capacity(selectors.dims);
    for dimension in 0..selectors.dims {
        if selectors.colon_mask & (1_u32 << dimension) != 0 {
            components.push(ObjectIndexComponent::Colon);
        } else {
            let value = values.next().ok_or_else(|| {
                NativeExecutorError::Host("object selector is missing a component".into())
            })?;
            components.push(ObjectIndexComponent::Value(value.clone()));
        }
    }
    if values.next().is_some() {
        return Err(NativeExecutorError::Host(
            "object selector contains excess components".into(),
        ));
    }
    Ok(ObjectIndexSelector::IndexValues { components })
}

pub(super) fn read(
    state: &mut HostState,
    base: &MirOperand,
    indexing: &MirIndexing,
    requested_outputs: usize,
) -> NativeExecutorResult<runmat_value::OutputListStorage> {
    let embedded = state.enter_embedded_call()?;
    if let Some(values) = super::call_suspension::take_completed_semantic_operation(
        state,
        embedded.as_ref(),
        super::call_suspension::SemanticOperationKind::IndexedRead,
    )? {
        return super::call_suspension::settle_completed_semantic_values(
            state,
            embedded.as_ref(),
            values,
        );
    }
    let result = read_inner(state, base, indexing, requested_outputs, embedded.as_ref());
    state
        .settle_embedded_result(embedded.as_ref(), result, IndexedReadOutcome::values)
        .map(IndexedReadOutcome::into_storage)
}

fn read_inner(
    state: &mut HostState,
    base: &MirOperand,
    indexing: &MirIndexing,
    requested_outputs: usize,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<IndexedReadOutcome> {
    if let Some(outputs) = super::call_suspension::take_completed(state, embedded)? {
        return Ok(IndexedReadOutcome::Ordinary(
            runmat_value::OutputListStorage::unbounded(outputs),
        ));
    }
    let base = if indexing_requires_retention(indexing) {
        materialize_retained_operand(state, base)?.1
    } else {
        materialize_operand(state, base)?
    };
    read_value_inner(state, base, indexing, requested_outputs, embedded)
}

pub(super) fn read_value(
    state: &mut HostState,
    base: Value,
    indexing: &MirIndexing,
    requested_outputs: usize,
) -> NativeExecutorResult<runmat_value::OutputListStorage> {
    let embedded = state.enter_embedded_call()?;
    if let Some(values) = super::call_suspension::take_completed_semantic_operation(
        state,
        embedded.as_ref(),
        super::call_suspension::SemanticOperationKind::IndexedRead,
    )? {
        return super::call_suspension::settle_completed_semantic_values(
            state,
            embedded.as_ref(),
            values,
        );
    }
    let result = read_value_inner_with_completion(
        state,
        base,
        indexing,
        requested_outputs,
        embedded.as_ref(),
    );
    state
        .settle_embedded_result(embedded.as_ref(), result, IndexedReadOutcome::values)
        .map(IndexedReadOutcome::into_storage)
}

fn read_value_inner_with_completion(
    state: &mut HostState,
    base: Value,
    indexing: &MirIndexing,
    requested_outputs: usize,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<IndexedReadOutcome> {
    if let Some(outputs) = super::call_suspension::take_completed(state, embedded)? {
        return Ok(IndexedReadOutcome::Ordinary(
            runmat_value::OutputListStorage::unbounded(outputs),
        ));
    }
    read_value_inner(state, base, indexing, requested_outputs, embedded)
}

fn read_value_inner(
    state: &mut HostState,
    base: Value,
    indexing: &MirIndexing,
    requested_outputs: usize,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<IndexedReadOutcome> {
    // Owned-base reads are called only by mutation continuation `read_step`.
    // Its PendingPlacePrepare owns and roots the original arena reference
    // before selector preparation can suspend, so retaining another Value
    // here would add a non-canonical clone without strengthening identity.
    let retains_inputs = indexing_requires_retention(indexing);
    let selectors = materialize_selectors(state, &base, indexing, retains_inputs)?;
    match indexing.kind {
        IndexKind::Paren => {
            read_operation::read_paren(state, base, selectors, requested_outputs, embedded)
        }
        IndexKind::Brace => read_brace(base, selectors, indexing, requested_outputs)
            .map(IndexedReadOutcome::Ordinary),
    }
}

pub(super) fn assign(
    state: &mut HostState,
    base: Value,
    indexing: &MirIndexing,
    rhs: Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    let embedded = state.enter_embedded_call()?;
    if let Some(values) = super::call_suspension::take_completed_semantic_operation(
        state,
        embedded.as_ref(),
        super::call_suspension::SemanticOperationKind::IndexedAssignment,
    )? {
        let values = super::call_suspension::settle_completed_semantic_values(
            state,
            embedded.as_ref(),
            values,
        )?;
        if values.as_slice().len() != 1 {
            return Err(NativeExecutorError::Host(
                "completed indexed assignment did not produce one receiver".into(),
            ));
        }
        return super::call_suspension::own_output_storage(values)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                NativeExecutorError::Host("completed indexed assignment lost its receiver".into())
            });
    }
    let result =
        assign_inner_with_completion(state, base, indexing, rhs, delete, embedded.as_ref());
    state
        .settle_embedded_result(embedded.as_ref(), result, IndexedAssignmentOutcome::values)
        .and_then(IndexedAssignmentOutcome::into_value)
}

fn assign_inner_with_completion(
    state: &mut HostState,
    base: Value,
    indexing: &MirIndexing,
    rhs: Value,
    delete: bool,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<IndexedAssignmentOutcome> {
    if let Some(mut values) = super::call_suspension::take_completed(state, embedded)? {
        if values.len() != 1 {
            return Err(NativeExecutorError::Host(
                "completed object assignment did not produce one receiver".into(),
            ));
        }
        return Ok(IndexedAssignmentOutcome::Ordinary(values.remove(0)));
    }
    // PendingAssign owns and roots both the receiver and RHS before write_step
    // reaches selector preparation; a suspended selector resumes from those
    // exact references rather than rereading either source binding.
    let retains_inputs = indexing_requires_retention(indexing);
    let selectors = materialize_selectors(state, &base, indexing, retains_inputs)?;
    if indexing.kind == IndexKind::Brace {
        return assign_brace(base, selectors, rhs, delete).map(IndexedAssignmentOutcome::Ordinary);
    }
    if matches!(
        base,
        Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)
    ) {
        let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
        if let Some(method) = resolution {
            let path = ObjectSubscriptPath::single(ObjectSubscript::parentheses(object_selector(
                &selectors,
            )?));
            let values = runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs.clone())?
                .into_output_storage();
            let values = values
                .try_into_owned()
                .map_err(|_| {
                    NativeExecutorError::from(semantic_error(
                        "ManagedObjectAssignmentOutputUnavailable",
                        "object assignment cannot detach shared managed output storage",
                    ))
                })?
                .try_into_unbounded_vec()
                .map_err(|_| {
                    NativeExecutorError::from(semantic_error(
                        "ManagedObjectAssignmentOutputUnavailable",
                        "object assignment requires an unbounded output owner",
                    ))
                })?;
            let roots = RootedValueLease::for_values(
                std::iter::once(&base)
                    .chain(selectors.numeric.iter())
                    .chain(std::iter::once(&rhs)),
            )
            .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
            let runtime = state.runtime.clone();
            let mut outputs = super::call_suspension::begin_rooted_arguments(
                state,
                embedded.cloned(),
                Box::pin(async move {
                    let _roots = roots;
                    let value = runtime
                        .scope(runmat_runtime::object::protocol::invoke_resolved_object_assignment_in_context(
                            &runtime, &method, base, path, values,
                        ))
                        .await?;
                    value.with_owned(|value| {
                        RootedValues::new(vec![value]).map_err(|error| error.into_runtime_error())
                    })
                }),
            )?;
            if outputs.len() != 1 {
                return Err(NativeExecutorError::Host(
                    "object assignment did not produce one receiver".into(),
                ));
            }
            return Ok(IndexedAssignmentOutcome::Ordinary(outputs.remove(0)));
        }
    }
    let direct_host = assignment_operation::host_assignment_eligible(&base, &rhs)
        && direct_host_indexing_allowed(state)?;
    let roots = RootedValueLease::for_values(
        std::iter::once(&base)
            .chain(selectors.numeric.iter())
            .chain(std::iter::once(&rhs)),
    )
    .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    if direct_host {
        if let Some(plan) =
            assignment_operation::plan_for_assignment_host(&base, &selectors, !delete)?
        {
            let updated = assignment_operation::assign_host_with_plan(base, &plan, rhs, delete)?;
            let rooted = RootedValues::in_context(vec![updated], &state.runtime)
                .map_err(NativeExecutorError::from)?;
            drop(roots);
            return Ok(IndexedAssignmentOutcome::Host(rooted));
        }
    }
    let runtime = state.runtime.clone();
    let future = async move {
        let _roots = roots;
        let updated = runtime
            .scope(assignment_operation::assign_paren_owned(
                base, selectors, rhs, delete,
            ))
            .await?;
        RootedValues::in_context(vec![updated], &runtime).map_err(NativeExecutorError::from)
    };
    let values = super::call_suspension::begin_semantic_operation(
        state,
        embedded.cloned(),
        super::call_suspension::SemanticOperationKind::IndexedAssignment,
        Vec::new(),
        future,
    )?;
    values.with_owned_output_storage(|values| {
        if values.as_slice().len() != 1 {
            return Err(NativeExecutorError::Host(
                "indexed assignment did not produce one receiver".into(),
            ));
        }
        super::call_suspension::own_output_storage(values)?
            .into_iter()
            .next()
            .map(IndexedAssignmentOutcome::Ordinary)
            .ok_or_else(|| NativeExecutorError::Host("indexed assignment lost its receiver".into()))
    })
}

struct MaterializedSelectors {
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: Vec<Value>,
    positional: Vec<Value>,
}

fn materialize_selectors(
    state: &mut HostState,
    base: &Value,
    indexing: &MirIndexing,
    retains_inputs: bool,
) -> NativeExecutorResult<MaterializedSelectors> {
    if indexing.components.len() > u32::BITS as usize {
        return Err(NativeExecutorError::Host(
            "verified Native IR exceeds the shared 32-dimension selector mask".into(),
        ));
    }
    let shape = value_shape(base);
    let dims = indexing.components.len();
    let mut colon_mask = 0_u32;
    let end_mask = 0_u32;
    let mut numeric = Vec::new();
    let mut positional = Vec::with_capacity(dims);
    for (dimension, component) in indexing.components.iter().enumerate() {
        match component {
            MirIndexComponent::Colon => {
                colon_mask |= 1_u32 << dimension;
                positional.push(Value::String(":".into()));
            }
            MirIndexComponent::Expr(operand) => {
                let value = if retains_inputs {
                    materialize_retained_operand(state, operand)?.1
                } else {
                    materialize_operand(state, operand)?
                };
                numeric.push(value.clone());
                positional.push(value);
            }
            MirIndexComponent::PreparedExpr(region) => {
                let value = materialize_prepared_expression_region(state, region)?;
                numeric.push(value.clone());
                positional.push(value);
            }
            MirIndexComponent::ContextualExpr(region) => {
                let extent = selector_dimension_length(&shape, dims, dimension)?;
                let value = materialize_expression_region(state, region, extent)?;
                numeric.push(value.clone());
                positional.push(value);
            }
        }
    }
    Ok(MaterializedSelectors {
        dims,
        colon_mask,
        end_mask,
        numeric,
        positional,
    })
}

pub(super) fn indexing_requires_retention(indexing: &MirIndexing) -> bool {
    indexing.components.iter().any(|component| {
        matches!(
            component,
            MirIndexComponent::ContextualExpr(_) | MirIndexComponent::PreparedExpr(_)
        )
    })
}

pub(super) fn materialize_index_components(
    state: &mut HostState,
    base: &Value,
    indexing: &MirIndexing,
) -> NativeExecutorResult<Vec<ObjectIndexComponent>> {
    let selectors =
        materialize_selectors(state, base, indexing, indexing_requires_retention(indexing))?;
    let mut values = selectors.positional.into_iter();
    indexing
        .components
        .iter()
        .map(|component| match component {
            MirIndexComponent::Colon => {
                let _ = values.next();
                Ok(ObjectIndexComponent::Colon)
            }
            MirIndexComponent::Expr(_)
            | MirIndexComponent::PreparedExpr(_)
            | MirIndexComponent::ContextualExpr(_) => values
                .next()
                .map(ObjectIndexComponent::Value)
                .ok_or_else(|| {
                    NativeExecutorError::Host(
                        "materialized index component is missing its value".into(),
                    )
                }),
        })
        .collect()
}

pub(super) fn materialize_cell_expansion_source(
    state: &mut HostState,
    base: Value,
    indexing: &MirIndexing,
) -> NativeExecutorResult<runmat_runtime::call::arguments::MaterializedExpansionSource> {
    let retains_inputs = indexing_requires_retention(indexing);
    let selectors = materialize_selectors(state, &base, indexing, retains_inputs)?;
    Ok(
        runmat_runtime::call::arguments::MaterializedExpansionSource::CellContents {
            base,
            indices: if indexing.cell_expand_all {
                Vec::new()
            } else {
                selectors.positional
            },
            expand_all: indexing.cell_expand_all,
        },
    )
}

pub(super) fn materialize_expression_region(
    state: &mut HostState,
    region: &runmat_mir::MirExpressionRegion,
    extent: usize,
) -> NativeExecutorResult<Value> {
    region.validate().map_err(|error| {
        NativeExecutorError::Host(format!("invalid contextual selector: {error}"))
    })?;
    let (ordinal, next_step, completed) = state.enter_contextual_region()?;
    if let Some(result) = completed {
        let [result] = result.as_slice() else {
            return Err(NativeExecutorError::Host(
                "contextual selector completed with non-scalar result".into(),
            ));
        };
        return state.arena.get(*result).cloned();
    }
    state.contextual_index_extents.push(extent);
    let result = (|| {
        execute_preparation_steps(state, ordinal, region.steps(), next_step)?;
        let result = super::operand::evaluate_operand(state, region.result())?;
        state.retain_expression_region_result(result)?;
        state.finish_contextual_region(ordinal, result);
        state.arena.get(result).cloned()
    })();
    state.contextual_index_extents.pop();
    if result.is_err() && !matches!(&result, Err(NativeExecutorError::CallSuspended)) {
        state.abandon_contextual_region(region.defined_sequences())?;
    }
    result
}

pub(super) fn materialize_prepared_expression_region(
    state: &mut HostState,
    region: &runmat_mir::MirPreparedExpressionRegion,
) -> NativeExecutorResult<Value> {
    region.validate().map_err(|error| {
        NativeExecutorError::Host(format!("invalid prepared expression region: {error}"))
    })?;
    materialize_expression_region_body(
        state,
        region.steps(),
        region.result(),
        region.defined_sequences(),
    )
}

fn materialize_expression_region_body<'a>(
    state: &mut HostState,
    steps: &[runmat_mir::MirExpressionStep],
    result_operand: &runmat_mir::MirOperand,
    defined_sequences: impl Iterator<Item = runmat_mir::MirSequenceLocalId> + 'a,
) -> NativeExecutorResult<Value> {
    let (ordinal, next_step, completed) = state.enter_contextual_region()?;
    if let Some(result) = completed {
        let [result] = result.as_slice() else {
            return Err(NativeExecutorError::Host(
                "prepared expression region completed with non-scalar result".into(),
            ));
        };
        return state.arena.get(*result).cloned();
    }
    let result = (|| {
        execute_preparation_steps(state, ordinal, steps, next_step)?;
        let result = super::operand::evaluate_operand(state, result_operand)?;
        state.retain_expression_region_result(result)?;
        state.finish_contextual_region(ordinal, result);
        state.arena.get(result).cloned()
    })();
    if result.is_err() && !matches!(&result, Err(NativeExecutorError::CallSuspended)) {
        state.abandon_contextual_region(defined_sequences)?;
    }
    result
}

pub(super) fn execute_preparation_steps(
    state: &mut HostState,
    ordinal: usize,
    steps: &[runmat_mir::MirExpressionStep],
    next_step: usize,
) -> NativeExecutorResult<()> {
    for (step_index, step) in steps.iter().enumerate().skip(next_step) {
        state.enter_contextual_step(ordinal, step_index)?;
        match step {
            runmat_mir::MirExpressionStep::Let { local, value, .. } => {
                let native_local = u32::try_from(local.0)
                    .map(runmat_native_codegen::NativeLocalId)
                    .map_err(|_| {
                        NativeExecutorError::Host(
                            "expression preparation local exceeds native schema".into(),
                        )
                    })?;
                let mut values =
                    super::operand::evaluate_rvalue(state, value, 1, Some(native_local))?;
                if values.len() != 1 {
                    return Err(NativeExecutorError::Host(
                        "expression preparation did not produce one local value".into(),
                    ));
                }
                state.set_local(local.0, values.remove(0))?;
            }
            runmat_mir::MirExpressionStep::CaptureSequence {
                destination,
                source,
                ..
            } => {
                super::call::capture_sequence(state, *destination, source)?;
            }
        }
        state.finish_contextual_step(ordinal, step_index, step_index + 1)?;
    }
    Ok(())
}

fn selector_dimension_length(
    shape: &[usize],
    dims: usize,
    dimension: usize,
) -> NativeExecutorResult<usize> {
    runmat_runtime::indexing::shape::selector_extent(shape, dims, dimension)
        .map_err(NativeExecutorError::from)
}

fn read_brace(
    base: Value,
    selectors: MaterializedSelectors,
    indexing: &MirIndexing,
    requested_outputs: usize,
) -> NativeExecutorResult<runmat_value::OutputListStorage> {
    let Value::Cell(cell) = base else {
        return Err(NativeExecutorError::from(semantic_error(
            "CellIndexType",
            "Brace indexing requires a cell array",
        )));
    };
    let values = runmat_runtime::object::cell::expand_cell_indices(&cell, &selectors.positional)?;
    if indexing.result_context == IndexResultContext::ReadCommaList || indexing.cell_expand_all {
        if values.len() != requested_outputs {
            return Err(NativeExecutorError::from(semantic_error(
                "OutputArityMismatch",
                format!(
                    "cell comma-list produced {} values for {requested_outputs} outputs",
                    values.len()
                ),
            )));
        }
        return Ok(runmat_value::OutputListStorage::unbounded(values));
    }
    match values.as_slice() {
        [value] if requested_outputs == 1 => Ok(runmat_value::OutputListStorage::unbounded(vec![
            value.clone(),
        ])),
        _ => Err(NativeExecutorError::from(semantic_error(
            "CellIndexArity",
            "Cell brace indexing must select exactly one value in scalar context",
        ))),
    }
}

fn assign_brace(
    base: Value,
    selectors: MaterializedSelectors,
    rhs: Value,
    delete: bool,
) -> NativeExecutorResult<Value> {
    if delete {
        return Err(NativeExecutorError::from(semantic_error(
            "UnsupportedCellBraceDeletion",
            "Cell brace assignment does not support deletion",
        )));
    }
    let Value::Cell(cell) = base else {
        return Err(NativeExecutorError::from(semantic_error(
            "CellIndexType",
            "Brace assignment requires a cell array",
        )));
    };
    let positions = runmat_runtime::object::cell::resolve_cell_assignment_positions(
        &cell,
        &selectors.positional,
    )?;
    let sequence = runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?;
    let kind = sequence.kind();
    let values = match kind {
        runmat_value::ValueSequenceKind::CommaSeparated => sequence.into_output_storage(),
        runmat_value::ValueSequenceKind::Single if positions.len() == 1 => {
            sequence.into_output_storage()
        }
        runmat_value::ValueSequenceKind::Single => {
            let value = super::call_suspension::own_output_storage(sequence.into_output_storage())?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    semantic_error(
                        "CellAssignmentArityMismatch",
                        "single cell assignment value is missing",
                    )
                })?;
            runmat_value::OutputListStorage::unbounded(vec![value; positions.len()])
        }
    };
    runmat_runtime::object::cell::assign_cell_value_multi(
        cell,
        &positions,
        values.as_slice(),
        |_, _| {},
    )
    .map_err(NativeExecutorError::from)
}

fn read_with_plan(base: Value, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    match base {
        Value::Tensor(value) => {
            read_slice::read_tensor_slice_from_plan(&value, plan).map_err(NativeExecutorError::from)
        }
        Value::ComplexTensor(value) => read_slice::read_complex_slice_from_plan(&value, plan)
            .map_err(NativeExecutorError::from),
        Value::SparseTensor(value) => {
            read_slice::read_sparse_slice_from_plan(&value, plan).map_err(NativeExecutorError::from)
        }
        Value::GpuTensor(value) => {
            read_slice::read_gpu_slice_from_plan(&value, plan).map_err(NativeExecutorError::from)
        }
        Value::StringArray(value) => {
            read_slice::gather_string_slice(&value, plan).map_err(NativeExecutorError::from)
        }
        Value::LogicalArray(value) => gather_logical(&value, plan),
        value @ Value::CharArray(_) => runmat_runtime::indexing::value::read_with_plan(value, plan)
            .map_err(NativeExecutorError::from),
        Value::Cell(value) => {
            let indices = plan
                .indices
                .iter()
                .map(|index| *index as usize + 1)
                .collect::<Vec<_>>();
            runmat_runtime::object::cell::gather_cell_paren_linear_indices(
                &value,
                &indices,
                &plan.output_shape,
            )
            .map_err(NativeExecutorError::from)
        }
        Value::StructArray(value) => {
            runmat_runtime::indexing::structure::read_with_plan(&value, plan)
                .map_err(NativeExecutorError::from)
        }
        value @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
            runmat_runtime::indexing::object::read_with_plan(&value, plan)
                .map_err(NativeExecutorError::from)
        }
        Value::SymbolicArray(value) => gather_symbolic(&value, plan),
        value => read_scalar(value, plan),
    }
}

fn gather_logical(value: &LogicalArray, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    let values = gather(&value.data, plan, "logical")?;
    if let [value] = values.as_slice() {
        return Ok(Value::Bool(*value != 0));
    }
    LogicalArray::new(values, plan.output_shape.clone())
        .map(Value::LogicalArray)
        .map_err(|error| NativeExecutorError::from(shape_error(error)))
}

fn gather_symbolic(value: &SymbolicArray, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    let values = gather(&value.data, plan, "symbolic")?;
    if let [value] = values.as_slice() {
        return Ok(Value::Symbolic(value.clone()));
    }
    SymbolicArray::new(values, plan.output_shape.clone())
        .map(Value::SymbolicArray)
        .map_err(|error| NativeExecutorError::from(shape_error(error)))
}

fn read_scalar(value: Value, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    if plan.indices.as_slice() == [0] {
        Ok(value)
    } else {
        Err(NativeExecutorError::from(semantic_error(
            "IndexOutOfBounds",
            "Index out of bounds",
        )))
    }
}

fn gather<T: Clone>(data: &[T], plan: &IndexPlan, kind: &str) -> NativeExecutorResult<Vec<T>> {
    plan.indices
        .iter()
        .map(|index| {
            data.get(*index as usize).cloned().ok_or_else(|| {
                NativeExecutorError::from(semantic_error(
                    "IndexOutOfBounds",
                    format!("{kind} index is out of bounds"),
                ))
            })
        })
        .collect()
}

fn normalize_outputs(
    value: Value,
    requested_outputs: usize,
) -> NativeExecutorResult<runmat_value::OutputListStorage> {
    let values =
        runmat_runtime::call::arguments::adapt_legacy_builtin_result(value)?.into_output_storage();
    normalize_output_values(values, requested_outputs).map_err(NativeExecutorError::Runtime)
}

fn normalize_sequence_outputs(
    sequence: runmat_value::ValueSequence,
    requested_outputs: usize,
) -> Result<runmat_value::OutputListStorage, Box<RuntimeError>> {
    normalize_output_values(sequence.into_output_storage(), requested_outputs)
}

fn normalize_output_values(
    values: runmat_value::OutputListStorage,
    requested_outputs: usize,
) -> Result<runmat_value::OutputListStorage, Box<RuntimeError>> {
    match (requested_outputs, values.as_slice().len()) {
        (0, _) => Ok(runmat_value::OutputListStorage::unbounded(Vec::new())),
        (expected, actual) if expected == actual => Ok(values),
        (expected, actual) => Err(Box::new(semantic_error(
            "OutputArityMismatch",
            format!("indexing produced {actual} values for {expected} outputs"),
        ))),
    }
}

fn value_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(value) => value.shape().to_vec(),
        Value::ComplexTensor(value) => value.shape().to_vec(),
        Value::SparseTensor(value) => value.shape(),
        Value::GpuTensor(value) => value.shape().to_vec(),
        Value::StringArray(value) => value.shape.clone(),
        Value::LogicalArray(value) => value.shape.to_vec(),
        Value::CharArray(value) => value.shape().to_vec(),
        Value::Cell(value) => value.shape.clone(),
        Value::StructArray(value) => value.shape().to_vec(),
        Value::ObjectArray(value) => value.shape().to_vec(),
        Value::SymbolicArray(value) => value.shape.clone(),
        _ => vec![1, 1],
    }
}

fn is_function_value(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

fn semantic_error(identifier: &str, message: impl Into<String>) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(identifier, message)
}

fn shape_error(error: impl std::fmt::Display) -> RuntimeError {
    semantic_error("ShapeMismatch", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::assignment_operation::plan_for_assignment;
    use super::{
        assign, read_value, read_with_plan, selector_dimension_length, MaterializedSelectors,
    };
    use crate::execute::region::observation::test_support::observed_state;
    use crate::region::placement_test_support::Probe;
    use runmat_mir::{MirConstant, MirIndexComponent, MirIndexPlan, MirIndexing, MirOperand};
    use runmat_runtime::indexing::plan::IndexPlan;
    use runmat_types::{IndexKind, IndexResultContext};
    use runmat_value::{CharArray, Tensor, Value};
    use std::rc::Rc;

    #[test]
    fn host_index_read_and_write_finish_without_pending_semantic_call() {
        runmat_gc::gc_test_context(|| {
            let mut state = observed_state(&Rc::new(Probe::default()));
            assert!(super::direct_host_indexing_allowed(&state).expect("unbounded host policy"));
            let indexing = MirIndexing {
                kind: IndexKind::Paren,
                plan: MirIndexPlan::Slice,
                components: vec![MirIndexComponent::Expr(MirOperand::Constant(
                    MirConstant::Number("2".into()),
                ))],
                result_context: IndexResultContext::ReadSingle,
                cell_expand_all: false,
            };
            let base = Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap());
            let read = read_value(&mut state, base.clone(), &indexing, 1)
                .expect("host selector read must finish directly");
            assert_eq!(read.as_slice(), &[Value::Num(20.0)]);
            assert!(state.pending_call.is_none());
            let updated = assign(&mut state, base, &indexing, Value::Num(99.0), false)
                .expect("host selector write must finish directly");
            let Value::Tensor(updated) = updated else {
                panic!("indexed write must preserve tensor class");
            };
            assert_eq!(updated.materialize_f64(), vec![10.0, 99.0, 30.0]);
            assert!(state.pending_call.is_none());
        });
    }

    #[test]
    fn native_numeric_scalar_planning_uses_shared_bounded_and_growth_modes() {
        let selectors = MaterializedSelectors {
            dims: 1,
            colon_mask: 0,
            end_mask: 0,
            numeric: vec![Value::Num(3.0)],
            positional: vec![Value::Num(3.0)],
        };
        for base in [
            Value::Num(9.0),
            Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
        ] {
            let plan = futures::executor::block_on(plan_for_assignment(&base, &selectors, true))
                .expect("scalar assignment must admit growth");
            assert_eq!(plan.base_shape, vec![1, 3]);
            assert!(
                futures::executor::block_on(plan_for_assignment(&base, &selectors, false)).is_err(),
                "scalar deletion must remain bounded"
            );
        }
    }

    #[test]
    fn native_end_extent_collapses_trailing_dimensions() {
        let shape = [2, 3, 4];
        assert_eq!(selector_dimension_length(&shape, 1, 0).unwrap(), 24);
        assert_eq!(selector_dimension_length(&shape, 2, 0).unwrap(), 2);
        assert_eq!(selector_dimension_length(&shape, 2, 1).unwrap(), 12);
        assert_eq!(selector_dimension_length(&shape, 3, 2).unwrap(), 4);
    }

    #[test]
    fn native_character_slices_use_shared_column_major_index_plan() {
        let chars = CharArray::new("abcdwxyz".chars().collect(), 2, 4).expect("character array");
        let row = IndexPlan::new(vec![1, 3, 5, 7], vec![1, 4], vec![1, 4], 2, vec![2, 4]);
        let column = IndexPlan::new(vec![4, 5], vec![2, 1], vec![2, 1], 2, vec![2, 4]);
        let Value::CharArray(row) =
            read_with_plan(Value::CharArray(chars.clone()), &row).expect("row slice")
        else {
            panic!("expected character row");
        };
        assert_eq!(row.shape(), &[1, 4]);
        assert_eq!(row.row_string().as_deref(), Some("wxyz"));

        let Value::CharArray(column) =
            read_with_plan(Value::CharArray(chars), &column).expect("column slice")
        else {
            panic!("expected character column");
        };
        assert_eq!(column.shape(), &[2, 1]);
        assert_eq!(column.data, vec!['c', 'y']);
    }
}

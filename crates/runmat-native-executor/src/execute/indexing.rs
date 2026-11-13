use runmat_mir::{MirIndexComponent, MirIndexing, MirOperand};
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
};
use runmat_runtime::execution::{RootedValueLease, RootedValues};
use runmat_runtime::indexing::plan::IndexPlan;
use runmat_runtime::object::dispatch::resolve_object_index_protocol;
use runmat_runtime::object::indexing::{
    ObjectIndexComponent, ObjectIndexOp, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use runmat_runtime::RuntimeError;
use runmat_types::IndexKind;
use runmat_value::{SymbolicArray, Value};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::{materialize_operand, materialize_retained_operand};
use super::state::HostState;

mod assignment_operation;
mod brace_operation;
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

fn object_selector(
    runtime: &runmat_runtime::context::RuntimeContext,
    selectors: &MaterializedSelectors,
) -> NativeExecutorResult<ObjectIndexSelector> {
    if selectors.end_mask != 0 {
        return Err(semantic_error(
            "ContextualEndRequiresSubscriptPath",
            "object end must be evaluated through the contextual subscript path",
        )
        .into());
    }
    let mut components =
        runmat_runtime::object::indexing::prepare_components_in_context(runtime, selectors.dims)?;
    let mut values = selectors.numeric.values().iter();
    for dimension in 0..selectors.dims {
        let component = if selectors.colon_mask & (1_u32 << dimension) != 0 {
            ObjectIndexComponent::Colon
        } else {
            ObjectIndexComponent::Value(
                values
                    .next()
                    .ok_or_else(|| {
                        NativeExecutorError::Host("object selector is missing a component".into())
                    })?
                    .clone(),
            )
        };
        components.try_push(component).map_err(|_| {
            NativeExecutorError::Host(
                "object selector exceeded its admitted component width".into(),
            )
        })?;
    }
    if values.next().is_some() {
        return Err(NativeExecutorError::Host(
            "object selector contains excess components".into(),
        ));
    }
    Ok(ObjectIndexSelector::IndexValues {
        components: components.try_finish().map_err(|_| {
            NativeExecutorError::Host(
                "object selector did not fill its admitted component width".into(),
            )
        })?,
    })
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
        IndexKind::Paren => read_operation::read_paren(
            state,
            base,
            selectors,
            requested_outputs,
            embedded,
            indexing,
        ),
        IndexKind::Brace => brace_operation::read_brace(
            &state.runtime,
            base,
            selectors,
            indexing,
            requested_outputs,
        )
        .map(IndexedReadOutcome::Host),
    }
}

pub(in crate::execute) fn index_callsite(
    state: &HostState,
    indexing: &MirIndexing,
) -> NativeExecutorResult<Option<runmat_runtime::callsite::CallsiteInfo>> {
    let Some(spans) = indexing.argument_spans.as_deref() else {
        return Ok(None);
    };
    let source = state
        .runtime
        .program_source_id(state.function.source)
        .ok_or_else(|| {
            NativeExecutorError::Host(
                "native selector source is absent from invocation binding".into(),
            )
        })?;
    runmat_runtime::callsite::CallsiteInfo::from_borrowed_in_context(
        Some(source),
        spans,
        &state.runtime,
    )
    .map(Some)
    .map_err(Into::into)
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
        return brace_operation::assign_brace(&state.runtime, base, selectors, rhs, delete)
            .map(IndexedAssignmentOutcome::Ordinary);
    }
    if matches!(
        base,
        Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)
    ) {
        let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsasgn, None)?;
        if let Some(method) = resolution {
            let path = ObjectSubscriptPath::single_in_context(
                Some(&state.runtime),
                ObjectSubscript::parentheses(object_selector(&state.runtime, &selectors)?),
            )?;
            let runtime = state.runtime.clone();
            let roots = RootedValueLease::for_values_iterator_in_context(
                [&base, &rhs].into_iter(),
                &runtime,
            )?;
            return super::call_suspension::begin_admitted_semantic_operation_with(
                state, embedded.cloned(),
                super::call_suspension::SemanticOperationKind::IndexedAssignment,
                || async move {
                    let _roots = roots;
                    let _selector_roots = selectors.numeric;
                    let rhs = runmat_runtime::call::arguments::adapt_legacy_builtin_result(rhs)?;
                    let value = runtime.scope(
                        runmat_runtime::object::protocol::invoke_resolved_object_assignment_sequence_in_context(
                            &runtime, &method, base, path, rhs)).await?;
                    value.with_owned(|value| {
                        let mut value = Some(value);
                        RootedValues::try_initialize_in_context(1, &runtime, |_|
                            Ok::<_, NativeExecutorError>(value.take().expect("one assigned receiver")))
                    })
                }).map(IndexedAssignmentOutcome::Host);
        }
    }
    let direct_host = assignment_operation::host_assignment_eligible(&base, &rhs, delete)
        && direct_host_indexing_allowed(state)?;
    let roots = RootedValueLease::for_values(
        std::iter::once(&base)
            .chain(selectors.numeric.values().iter())
            .chain(std::iter::once(&rhs)),
    )
    .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    if direct_host {
        if let Some(plan) =
            assignment_operation::plan_for_assignment_host(&base, &selectors, !delete)?
        {
            let updated = assignment_operation::assign_host_with_plan(
                Some(&state.runtime),
                base,
                &plan,
                rhs,
                delete,
            )?;
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
                &runtime, base, selectors, rhs, delete,
            ))
            .await?;
        RootedValues::in_context(vec![updated], &runtime).map_err(NativeExecutorError::from)
    };
    let values = super::call_suspension::begin_semantic_operation(
        state,
        embedded.cloned(),
        super::call_suspension::SemanticOperationKind::IndexedAssignment,
        Vec::new(),
        move || future,
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
    numeric: RootedValues,
}

impl MaterializedSelectors {
    fn positional_in_context(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> NativeExecutorResult<RootedValues> {
        let mut values = self.numeric.values().iter();
        RootedValues::try_initialize_in_context(self.dims, runtime, |dimension| {
            if self.colon_mask & (1_u32 << dimension) != 0 {
                return runmat_runtime::call::argument_text::copy_str(":", Some(runtime))
                    .map(|text| Value::String(text.into()))
                    .map_err(Into::into);
            }
            values.next().cloned().ok_or_else(|| {
                NativeExecutorError::Host(
                    "positional selector is missing its rooted numeric value".into(),
                )
            })
        })
    }
}

fn materialize_selectors(
    state: &mut HostState,
    base: &Value,
    indexing: &MirIndexing,
    retains_inputs: bool,
) -> NativeExecutorResult<MaterializedSelectors> {
    if indexing
        .components
        .iter()
        .any(|component| matches!(component, MirIndexComponent::Expansion(_)))
    {
        return Err(NativeExecutorError::UnsupportedSite(
            "expanded parentheses require the ordered subscript-path executor".into(),
        ));
    }
    if indexing.components.len() > u32::BITS as usize {
        return Err(NativeExecutorError::Host(
            "verified Native IR exceeds the shared 32-dimension selector mask".into(),
        ));
    }
    let shape = value_shape(base);
    let dims = indexing.components.len();
    let colon_mask =
        indexing
            .components
            .iter()
            .enumerate()
            .fold(0u32, |mask, (dimension, component)| {
                if matches!(component, MirIndexComponent::Colon) {
                    mask | (1_u32 << dimension)
                } else {
                    mask
                }
            });
    let count = dims - colon_mask.count_ones() as usize;
    let runtime = state.runtime.clone();
    let mut components = indexing
        .components
        .iter()
        .enumerate()
        .filter(|(_, component)| !matches!(component, MirIndexComponent::Colon));
    // Plain operands remain arena-owned. With regions, retained input slots and
    // each region's retained result keep every earlier child rooted while later
    // initializer calls may execute or suspend. This is final backing, not a
    // temporary payload collector.
    let numeric = RootedValues::try_initialize_in_context(count, &runtime, |_| {
        let (dimension, component) = components.next().expect("precounted non-colon component");
        match component {
            MirIndexComponent::Expr(operand) if retains_inputs => {
                materialize_retained_operand(state, operand).map(|(_, value)| value)
            }
            MirIndexComponent::Expr(operand) => materialize_operand(state, operand),
            MirIndexComponent::PreparedExpr(region) => {
                materialize_prepared_expression_region(state, region)
            }
            MirIndexComponent::ContextualExpr(region) => {
                let extent = selector_dimension_length(&shape, dims, dimension)?;
                materialize_expression_region(state, region, extent)
            }
            _ => unreachable!("expanded and colon components excluded before admission"),
        }
    })?;
    Ok(MaterializedSelectors {
        dims,
        colon_mask,
        end_mask: 0,
        numeric,
    })
}

pub(super) fn indexing_requires_retention(indexing: &MirIndexing) -> bool {
    indexing.components.iter().any(|component| {
        matches!(
            component,
            MirIndexComponent::ContextualExpr(_)
                | MirIndexComponent::PreparedExpr(_)
                | MirIndexComponent::Expansion(_)
        )
    })
}

pub(super) fn materialize_index_components(
    state: &mut HostState,
    base: &Value,
    indexing: &MirIndexing,
) -> NativeExecutorResult<runmat_value::ExactArrayStorage<ObjectIndexComponent>> {
    if indexing
        .components
        .iter()
        .any(|component| matches!(component, MirIndexComponent::Expansion(_)))
    {
        return super::subscript_path::materialize_target_selectors(state, base, indexing);
    }
    let selectors =
        materialize_selectors(state, base, indexing, indexing_requires_retention(indexing))?;
    object_selector(&state.runtime, &selectors).map(|selector| {
        let ObjectIndexSelector::IndexValues { components } = selector else {
            unreachable!("canonical component selector")
        };
        components
    })
}

pub(super) fn materialize_cell_expansion_source(
    state: &mut HostState,
    base: Value,
    indexing: &MirIndexing,
) -> NativeExecutorResult<runmat_runtime::call::arguments::MaterializedExpansionSource> {
    let selectors = materialize_selectors(
        state,
        &base,
        indexing,
        indexing_requires_retention(indexing),
    )?;
    let indices = if indexing.cell_expand_all {
        runmat_value::OutputListStorage::empty()
    } else {
        selectors
            .positional_in_context(&state.runtime)?
            .with_owned_output_storage(|values| values)
    };
    Ok(
        runmat_runtime::call::arguments::MaterializedExpansionSource::CellContents {
            base,
            indices,
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

fn read_with_plan(
    runtime: Option<&runmat_runtime::context::RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
) -> NativeExecutorResult<Value> {
    match base {
        value @ (Value::Tensor(_)
        | Value::ComplexTensor(_)
        | Value::SparseTensor(_)
        | Value::GpuTensor(_)
        | Value::StringArray(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Cell(_)
        | Value::StructArray(_)
        | Value::Object(_)
        | Value::ObjectArray(_)
        | Value::HandleObject(_)) => {
            runmat_runtime::indexing::value::read_with_plan_in_context(runtime, value, plan)
                .map_err(NativeExecutorError::from)
        }
        Value::SymbolicArray(value) => gather_symbolic(&value, plan),
        value => read_scalar(value, plan),
    }
}

fn gather_symbolic(value: &SymbolicArray, plan: &IndexPlan) -> NativeExecutorResult<Value> {
    let values = gather(value.data(), plan, "symbolic")?;
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

fn value_shape(value: &Value) -> runmat_runtime::builtins::common::shape::ValueDimensions<'_> {
    runmat_runtime::builtins::common::shape::stored_dimensions_borrowed(value)
}

fn is_function_value(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
            | Value::ScopedFunctionHandle(_)
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
    use runmat_runtime::execution::RootedValues;
    use runmat_runtime::indexing::plan::IndexPlan;
    use runmat_types::{IndexKind, IndexResultContext};
    use runmat_value::{CharArray, Tensor, Value};
    use std::rc::Rc;

    #[test]
    fn host_index_read_and_write_finish_without_pending_semantic_call() {
        let logical =
            Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap());
        assert!(super::assignment_operation::host_assignment_eligible(
            &logical,
            &Value::Num(1.),
            false
        ));
        assert!(!super::assignment_operation::host_assignment_eligible(
            &logical,
            &Value::Num(1.),
            true
        ));
        runmat_gc::gc_test_context(|| {
            let mut state = observed_state(&Rc::new(Probe::default()));
            assert!(super::direct_host_indexing_allowed(&state).expect("unbounded host policy"));
            let indexing = MirIndexing {
                argument_spans: None,
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
            let sparse_indexing = MirIndexing {
                components: vec![
                    MirIndexComponent::Colon,
                    MirIndexComponent::Expr(MirOperand::Constant(MirConstant::Number("1".into()))),
                ],
                ..indexing
            };
            let sparse = runmat_value::SparseTensor::new(
                1_000_000_000,
                1,
                vec![0, 1],
                vec![999_999_999],
                vec![2.0],
            )
            .unwrap();
            let output = read_value(&mut state, Value::SparseTensor(sparse), &sparse_indexing, 1)
                .expect("sparse host colon must not build dense coordinate metadata");
            let [Value::SparseTensor(output)] = output.as_slice() else {
                panic!("sparse host output");
            };
            assert_eq!(output.shape(), [1_000_000_000, 1]);
            assert_eq!(output.row_indices.as_ref(), &[999_999_999]);
            assert_eq!(output.as_f64_slice(), Some(&[2.0][..]));
            assert!(state.pending_call.is_none());
        });
    }

    #[test]
    fn native_numeric_scalar_planning_uses_shared_bounded_and_growth_modes() {
        let selectors = MaterializedSelectors {
            dims: 1,
            colon_mask: 0,
            end_mask: 0,
            numeric: RootedValues::new(vec![Value::Num(3.0)]).expect("root fixture selectors"),
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
        let chars = CharArray::from_unicode_scalars("abcdwxyz".chars().collect(), 2, 4)
            .expect("character array");
        let row = IndexPlan::new(vec![1, 3, 5, 7], vec![1, 4], vec![1, 4], 2, vec![2, 4]);
        let column = IndexPlan::new(vec![4, 5], vec![2, 1], vec![2, 1], 2, vec![2, 4]);
        let Value::CharArray(row) =
            read_with_plan(None, Value::CharArray(chars.clone()), &row).expect("row slice")
        else {
            panic!("expected character row");
        };
        assert_eq!(row.shape(), &[1, 4]);
        assert_eq!(row.row_string().as_deref(), Some("wxyz"));

        let Value::CharArray(column) =
            read_with_plan(None, Value::CharArray(chars), &column).expect("column slice")
        else {
            panic!("expected character column");
        };
        assert_eq!(column.shape(), &[2, 1]);
        assert_eq!(column.data(), runmat_value::encode_utf16_code_units("cy"));
    }
}

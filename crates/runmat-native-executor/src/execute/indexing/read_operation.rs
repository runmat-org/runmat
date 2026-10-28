//! The resumable indexed-read operation and its canonical selector plan.

use runmat_runtime::execution::{RootedValueLease, RootedValues};
use runmat_runtime::indexing::plan::{build_index_plan, IndexPlan};
use runmat_runtime::indexing::read_slice::{
    read_sparse_slice_with_policy_in_context, try_read_sparse_slice_host_in_context,
    SparseReadZeroPolicy,
};
use runmat_runtime::indexing::selectors::{build_slice_selectors, build_slice_selectors_host};
use runmat_runtime::object::dispatch::{
    invoke_resolved_object_index_path_method_in_context, resolve_object_index_protocol,
};
use runmat_runtime::object::indexing::{ObjectIndexOp, ObjectSubscript, ObjectSubscriptPath};
use runmat_value::Value;

use crate::execute::call_suspension;
use crate::execute::state::{EmbeddedOperationIdentity, HostState};
use crate::{NativeExecutorError, NativeExecutorResult};

use super::{
    direct_host_indexing_allowed, is_function_value, normalize_outputs, normalize_sequence_outputs,
    object_selector, read_with_plan, semantic_error, value_shape, IndexedReadOutcome,
    MaterializedSelectors,
};

pub(super) fn read_paren(
    state: &mut HostState,
    base: Value,
    selectors: MaterializedSelectors,
    requested_outputs: usize,
    embedded: Option<&EmbeddedOperationIdentity>,
    indexing: &runmat_mir::MirIndexing,
) -> NativeExecutorResult<IndexedReadOutcome> {
    if is_function_value(&base) {
        if selectors.colon_mask != 0 || selectors.end_mask != 0 {
            return Err(NativeExecutorError::from(semantic_error(
                "UnsupportedFunctionHandleSelector",
                "Function handle call does not support colon or end selector syntax",
            )));
        }
        let roots = RootedValueLease::for_values_iterator_in_context(
            std::iter::once(&base),
            &state.runtime,
        )?;
        let runtime = state.runtime.clone();
        let input_callsite = super::index_callsite(state, indexing)?;
        let arguments = selectors.numeric;
        return call_suspension::begin_rooted_arguments(
            state,
            embedded.cloned(),
            move || async move {
                let _roots = roots;
                let sequence = runtime
                    .scope(
                        runmat_runtime::call_feval_retained_sequence_async_with_outputs_and_callsite_in_context(
                            &runtime,
                            base,
                            arguments.into(),
                            requested_outputs,
                            input_callsite,
                        ),
                    )
                    .await?;
                sequence.with_owned_sequence(|sequence| {
                    let values = normalize_sequence_outputs(sequence, requested_outputs)
                        .map_err(|error| *error)?;
                    RootedValues::from_output_storage_in_context(values, &runtime)
                })
            },
        )
        .map(|values| {
            IndexedReadOutcome::Ordinary(runmat_value::OutputListStorage::unbounded(values))
        });
    }
    if matches!(
        base,
        Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)
    ) {
        let resolution = resolve_object_index_protocol(&base, ObjectIndexOp::Subsref, None)?;
        if resolution.is_some() {
            let path = ObjectSubscriptPath::single_in_context(
                Some(&state.runtime),
                ObjectSubscript::parentheses(object_selector(&state.runtime, &selectors)?),
            )?;
            let roots = RootedValueLease::for_values_iterator_in_context(
                std::iter::once(&base),
                &state.runtime,
            )?;
            let selector_roots = selectors.numeric;
            let runtime = state.runtime.clone();
            return call_suspension::begin_rooted_arguments(
                state,
                embedded.cloned(),
                move || async move {
                    let _roots = roots;
                    let _selector_roots = selector_roots;
                    let sequence = runtime
                        .scope(invoke_resolved_object_index_path_method_in_context(
                            &runtime,
                            &resolution,
                            base,
                            path,
                            requested_outputs,
                        ))
                        .await?;
                    sequence.with_owned_sequence(|sequence| {
                        let values = normalize_sequence_outputs(sequence, requested_outputs)
                            .map_err(|error| *error)?;
                        RootedValues::from_output_storage_in_context(values, &runtime)
                    })
                },
            )
            .map(|values| {
                IndexedReadOutcome::Ordinary(runmat_value::OutputListStorage::unbounded(values))
            });
        }
    }
    let direct_host = host_read_eligible(&base) && direct_host_indexing_allowed(state)?;
    let roots =
        RootedValueLease::for_values_iterator_in_context(std::iter::once(&base), &state.runtime)?;
    if direct_host {
        if let Value::SparseTensor(sparse) = &base {
            if let Some(value) = try_read_sparse_slice_host_in_context(
                &state.runtime,
                sparse,
                selectors.dims,
                selectors.colon_mask,
                selectors.end_mask,
                selectors.numeric.values(),
                SparseReadZeroPolicy::PreserveTypedComponents,
            )? {
                let outputs = normalize_outputs(value, requested_outputs)?;
                let rooted = RootedValues::from_output_storage_in_context(outputs, &state.runtime)?;
                drop(roots);
                return Ok(IndexedReadOutcome::Host(rooted));
            }
        } else if let Some(plan) = plan_for_host(&base, &selectors)? {
            let value = read_with_plan(Some(&state.runtime), base, &plan)?;
            let outputs = normalize_outputs(value, requested_outputs)?;
            let rooted = RootedValues::from_output_storage_in_context(outputs, &state.runtime)
                .map_err(NativeExecutorError::from)?;
            drop(roots);
            return Ok(IndexedReadOutcome::Host(rooted));
        }
    }
    let runtime = state.runtime.clone();
    let future = async move {
        let _roots = roots;
        let outputs = runtime
            .scope(async {
                let value = if let Value::SparseTensor(sparse) = &base {
                    read_sparse_slice_with_policy_in_context(
                        Some(&runtime),
                        sparse,
                        selectors.dims,
                        selectors.colon_mask,
                        selectors.end_mask,
                        selectors.numeric.values(),
                        SparseReadZeroPolicy::PreserveTypedComponents,
                    )
                    .await?
                } else {
                    let plan = plan_for(&base, &selectors).await?;
                    read_with_plan(Some(&runtime), base, &plan)?
                };
                normalize_outputs(value, requested_outputs)
            })
            .await?;
        RootedValues::from_output_storage_in_context(outputs, &runtime)
            .map_err(NativeExecutorError::from)
    };
    let values = call_suspension::begin_semantic_operation(
        state,
        embedded.cloned(),
        call_suspension::SemanticOperationKind::IndexedRead,
        Vec::new(),
        move || future,
    )?;
    Ok(IndexedReadOutcome::Ordinary(
        values.with_owned_output_storage(|values| values),
    ))
}

fn host_read_eligible(base: &Value) -> bool {
    matches!(
        base,
        Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::SparseTensor(_)
            | Value::StringArray(_)
            | Value::LogicalArray(_)
            | Value::CharArray(_)
            | Value::Cell(_)
            | Value::Struct(_)
            | Value::StructArray(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::String(_)
    )
}

fn plan_for_host(
    base: &Value,
    selectors: &MaterializedSelectors,
) -> NativeExecutorResult<Option<IndexPlan>> {
    let shape = value_shape(base);
    let Some(built) = build_slice_selectors_host(
        selectors.dims,
        selectors.colon_mask,
        selectors.end_mask,
        selectors.numeric.values(),
        &shape,
    )?
    else {
        return Ok(None);
    };
    build_index_plan(&built, selectors.dims, &shape)
        .map(Some)
        .map_err(NativeExecutorError::from)
}

async fn plan_for(
    base: &Value,
    selectors: &MaterializedSelectors,
) -> NativeExecutorResult<IndexPlan> {
    let shape = value_shape(base);
    let built = build_slice_selectors(
        selectors.dims,
        selectors.colon_mask,
        selectors.end_mask,
        selectors.numeric.values(),
        &shape,
    )
    .await?;
    build_index_plan(&built, selectors.dims, &shape).map_err(NativeExecutorError::from)
}

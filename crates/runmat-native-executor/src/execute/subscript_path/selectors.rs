use super::super::operand::{materialize_operand, materialize_retained_operand};
use super::super::state::HostState;
use crate::{NativeExecutorError, NativeExecutorResult};
use runmat_mir::MirIndexComponent;
use runmat_runtime::execution::{RootedValueLease, RootedValues};
use runmat_runtime::object::indexing::{
    ObjectIndexComponent, ObjectSubscript, ObjectSubscriptPath,
};
use runmat_value::{ExactArrayStorage, Value};

mod receiver;
use receiver::prepare_receiver;

pub(super) fn prepare_dotted_receiver(
    state: &mut HostState,
    step: usize,
    root: Value,
    prefix: Option<ObjectSubscriptPath>,
    access: runmat_runtime::object::protocol::ObjectAccessContext,
) -> NativeExecutorResult<runmat_runtime::object::protocol::PreparedSubscriptReceiver> {
    prepare_receiver(state, step, root, prefix, access)
}

pub(super) struct MaterializedSelectors {
    pub components: ExactArrayStorage<ObjectIndexComponent>,
    pub prepared_receiver: Option<Value>,
}

pub(super) fn materialize_selectors(
    state: &mut HostState,
    step: usize,
    root: &Value,
    prefix: &[ObjectSubscript],
    indexing: &runmat_mir::MirIndexing,
    access: &runmat_runtime::object::protocol::ObjectAccessContext,
    retains_inputs: bool,
) -> NativeExecutorResult<MaterializedSelectors> {
    let runtime = state.runtime.clone();
    // One ordinary canonical rooted chunk per source component. This exact
    // control backing is admitted before End/prefix/selector effects; variable
    // CSL payloads keep their own existing rooted output storage.
    let mut chunks = runmat_runtime::context::host_allocation::prepare_host_array::<RootedValues>(
        &runtime,
        indexing.components.len(),
    )?;
    let needs_end = indexing.components.iter().any(|component|
        matches!(component, MirIndexComponent::ContextualExpr(region) if region.contains_contextual_end()));
    let prepared = needs_end
        .then(|| {
            let prefix = if prefix.is_empty() {
                None
            } else {
                let mut steps = runmat_runtime::object::indexing::prepare_subscripts_in_context(
                    &runtime,
                    prefix.len(),
                )?;
                for step in prefix {
                    steps.try_push(step.clone()).map_err(|_| {
                        NativeExecutorError::Host(
                            "prepared prefix exceeded its admitted width".into(),
                        )
                    })?;
                }
                Some(ObjectSubscriptPath::from_storage(
                    steps.try_finish().map_err(|_| {
                        NativeExecutorError::Host(
                            "prepared prefix did not fill its admitted width".into(),
                        )
                    })?,
                )?)
            };
            prepare_receiver(state, step, root.clone(), prefix, access.clone())
        })
        .transpose()?;
    let _roots = prepared
        .as_ref()
        .map(|prepared| {
            RootedValueLease::for_values_iterator_in_context(
                std::iter::once(prepared.receiver()),
                &runtime,
            )
        })
        .transpose()?;
    let mut width = 0usize;
    for (component, selector) in indexing.components.iter().enumerate() {
        let values = match selector {
            MirIndexComponent::Colon => RootedValues::try_initialize_in_context(
                0,
                &runtime,
                |_| -> NativeExecutorResult<Value> { unreachable!("colon has no numeric payload") },
            )?,
            MirIndexComponent::Expansion(region) => {
                super::arguments::materialize_call_region(state, region)?
            }
            _ => RootedValues::try_initialize_in_context(1, &runtime, |_| match selector {
                MirIndexComponent::Expr(operand) if retains_inputs => {
                    materialize_retained_operand(state, operand).map(|(_, value)| value)
                }
                MirIndexComponent::Expr(operand) => materialize_operand(state, operand),
                MirIndexComponent::PreparedExpr(region) => {
                    super::super::indexing::materialize_prepared_expression_region(state, region)
                }
                MirIndexComponent::ContextualExpr(region) => {
                    if let Some(prepared) = prepared.as_ref() {
                        state.subscript_end_receivers.push((
                            prepared.clone(),
                            component,
                            indexing.components.len(),
                        ));
                    }
                    let result =
                        super::super::indexing::materialize_expression_region(state, region, 1);
                    if prepared.is_some() {
                        state.subscript_end_receivers.pop();
                    }
                    result
                }
                _ => unreachable!("colon and expansion use their canonical chunk owners"),
            })?,
        };
        width = width
            .checked_add(if matches!(selector, MirIndexComponent::Colon) {
                1
            } else {
                values.values().len()
            })
            .ok_or_else(|| NativeExecutorError::Host("expanded selector width overflow".into()))?;
        chunks.try_push(values).map_err(|_| {
            NativeExecutorError::Host("selector chunks exceeded their admitted source width".into())
        })?;
    }
    let chunks = chunks.try_finish().map_err(|_| {
        NativeExecutorError::Host("selector chunks did not fill their admitted source width".into())
    })?;
    let mut components =
        runmat_runtime::object::indexing::prepare_components_in_context(&runtime, width)?;
    let chunks = chunks
        .try_into_owned()
        .map_err(|_| NativeExecutorError::Host("selector chunks lost unique ownership".into()))?;
    for (selector, values) in indexing.components.iter().zip(chunks) {
        if matches!(selector, MirIndexComponent::Colon) {
            components
                .try_push(ObjectIndexComponent::Colon)
                .map_err(|_| {
                    NativeExecutorError::Host("selector output exceeded admitted width".into())
                })?;
        } else {
            for value in values.try_into_values()? {
                components
                    .try_push(ObjectIndexComponent::Value(value))
                    .map_err(|_| {
                        NativeExecutorError::Host("selector output exceeded admitted width".into())
                    })?;
            }
        }
    }
    Ok(MaterializedSelectors {
        components: components.try_finish().map_err(|_| {
            NativeExecutorError::Host("selector output did not fill admitted width".into())
        })?,
        prepared_receiver: prepared.map(|prepared| prepared.receiver().clone()),
    })
}

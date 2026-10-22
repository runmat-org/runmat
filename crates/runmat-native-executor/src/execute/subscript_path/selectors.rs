use runmat_mir::MirIndexComponent;
use runmat_runtime::object::indexing::{
    ObjectIndexComponent, ObjectSubscript, ObjectSubscriptPath,
};
use runmat_value::Value;

use crate::memory::ScopedValueRoots;
use crate::NativeExecutorResult;

use super::super::operand::materialize_operand;
use super::super::operand::materialize_retained_operand;
use super::super::state::HostState;

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
    pub components: Vec<ObjectIndexComponent>,
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
    let needs_end = indexing.components.iter().any(|component| {
        matches!(component, MirIndexComponent::ContextualExpr(region) if region.contains_contextual_end())
    });
    let prepared = needs_end
        .then(|| {
            let prefix = if prefix.is_empty() {
                None
            } else {
                Some(ObjectSubscriptPath::new(prefix.to_vec())?)
            };
            prepare_receiver(state, step, root.clone(), prefix, access.clone())
        })
        .transpose()?;
    let _roots = prepared
        .as_ref()
        .map(|prepared| {
            ScopedValueRoots::register(
                vec![prepared.receiver().clone()],
                "native_subscript_end_receiver",
            )
        })
        .transpose()?;
    let mut components = Vec::new();
    for (component, selector) in indexing.components.iter().enumerate() {
        match selector {
            MirIndexComponent::Colon => components.push(ObjectIndexComponent::Colon),
            MirIndexComponent::Expr(operand) => {
                let value = if retains_inputs {
                    materialize_retained_operand(state, operand)?.1
                } else {
                    materialize_operand(state, operand)?
                };
                components.push(ObjectIndexComponent::Value(value));
            }
            MirIndexComponent::PreparedExpr(region) => {
                let value =
                    super::super::indexing::materialize_prepared_expression_region(state, region)?;
                components.push(ObjectIndexComponent::Value(value));
            }
            MirIndexComponent::Expansion(region) => {
                components.extend(
                    super::arguments::materialize_call_region(state, region)?
                        .into_iter()
                        .map(ObjectIndexComponent::Value),
                );
            }
            MirIndexComponent::ContextualExpr(region) => {
                if let Some(prepared) = prepared.as_ref() {
                    state.subscript_end_receivers.push((
                        prepared.clone(),
                        component,
                        indexing.components.len(),
                    ));
                }
                let value = super::super::indexing::materialize_expression_region(state, region, 1);
                if prepared.is_some() {
                    state.subscript_end_receivers.pop();
                }
                components.push(ObjectIndexComponent::Value(value?));
            }
        }
    }
    Ok(MaterializedSelectors {
        components,
        prepared_receiver: prepared.map(|prepared| prepared.receiver().clone()),
    })
}

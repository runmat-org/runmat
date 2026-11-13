//! Conservative selected-value output features of an exact Graph invocation.
//!
//! Only explicitly non-dispatching instructions are V10-closed. Every other
//! opcode, including newly added opcodes, requires V11 before effects; a
//! sealed graph input alone cannot prove what selected code may return.

use runmat_execution::value::{
    GraphBoundOwner, GraphCallableOwner, GraphInvocationRoots, GraphNode, GraphRequiredSchema,
    PayloadGraph,
};

use crate::Instr;

/// The instruction-level part of the selected-output requirement. This is a
/// positive V10 proof, not a list of known V11 constructors: unrecognized
/// execution semantics conservatively require the upgraded worker.
pub fn selected_graph_code_required_schema(instructions: &[Instr]) -> GraphRequiredSchema {
    if instructions.iter().all(|instruction| {
        matches!(
            instruction,
            Instr::LoadConst(_)
                | Instr::LoadInt(_)
                | Instr::LoadComplex(_, _)
                | Instr::LoadBool(_)
                | Instr::LoadString(_)
                | Instr::LoadCharRow(_)
                | Instr::LoadVar(_)
                | Instr::StoreVar(_)
                | Instr::Pop
                | Instr::Jump(_)
                | Instr::CreateCell2D(_, _)
                | Instr::CreateStructLiteral(_)
                | Instr::Return
                | Instr::ReturnValue
        )
    }) {
        GraphRequiredSchema::V10
    } else {
        GraphRequiredSchema::SelectedValuesV11
    }
}

/// Resolve the same typed executable root in Core and the worker. Each caller
/// supplies its own exact owner/body lookup; neither this helper nor either
/// closure may fall back to a numeric or textual current registry lookup.
pub fn selected_graph_execution_required_schema(
    roots: &GraphInvocationRoots,
    graph: &PayloadGraph,
    mut bound_body: impl FnMut(GraphBoundOwner) -> Result<GraphRequiredSchema, String>,
    script: impl FnOnce() -> Result<GraphRequiredSchema, String>,
) -> Result<GraphRequiredSchema, String> {
    let executable = match roots {
        GraphInvocationRoots::Callable { target, .. } => {
            selected_callable_root_schema(graph, *target, &mut bound_body)?
        }
        GraphInvocationRoots::ParallelRegion { .. } => script()?,
        GraphInvocationRoots::Builtin { name, .. } => {
            // Catalog identity is canonical and typed, unlike a display-name
            // cache. A pure builtin is V10-closed only without any transported
            // object or callable authority that could select constructor,
            // metadata, or callback code. Unknown opcodes above remain V11.
            let has_selected_authority_input = graph.nodes.iter().any(|node| {
                matches!(
                    node,
                    GraphNode::SelectedObject(_)
                        | GraphNode::SelectedObjectArray(_)
                        | GraphNode::SelectedDynamicPropertyMetadata(_)
                        | GraphNode::Callable(_)
                        | GraphNode::Activation(_)
                )
            });
            let v10_closed = !has_selected_authority_input
                && runmat_builtins::builtin_catalog_entry_by_name(&name.0).is_some_and(|entry| {
                    entry.identity.name == name.0
                        && entry.contract.purity == runmat_builtins::BuiltinPurity::Pure
                        && entry
                            .contract
                            .effects
                            .iter()
                            .all(|effect| *effect == runmat_types::EffectKind::MayThrow)
                });
            if v10_closed {
                GraphRequiredSchema::V10
            } else {
                GraphRequiredSchema::SelectedValuesV11
            }
        }
    };
    Ok(graph.required_schema().max(executable))
}

/// The same executable floor for each actual callable root in one portable
/// capture graph. Shared activation nodes remain one graph, not per-call copies.
pub fn selected_callable_capture_required_schema(
    graph: &PayloadGraph,
    mut bound_body: impl FnMut(GraphBoundOwner) -> Result<GraphRequiredSchema, String>,
) -> Result<GraphRequiredSchema, String> {
    graph
        .roots
        .iter()
        .try_fold(graph.required_schema(), |floor, root| {
            selected_callable_root_schema(graph, *root, &mut bound_body)
                .map(|schema| floor.max(schema))
        })
}

fn selected_callable_root_schema(
    graph: &PayloadGraph,
    target: runmat_execution::value::GraphNodeRef,
    bound_body: &mut impl FnMut(GraphBoundOwner) -> Result<GraphRequiredSchema, String>,
) -> Result<GraphRequiredSchema, String> {
    let Some(GraphNode::Callable(callable)) = graph.nodes.get(target.0 as usize) else {
        return Err("graph callable root is absent".into());
    };
    match callable.owner {
        GraphCallableOwner::Bound(owner) => bound_body(owner),
        // An unbound function name cannot certify selected source authority.
        _ => Ok(GraphRequiredSchema::SelectedValuesV11),
    }
}

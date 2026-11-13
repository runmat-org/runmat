use std::collections::{HashMap, HashSet};

use runmat_execution::value::{
    GraphBindingKind, GraphBoundOwner, GraphLexicalLayout, GraphProgramUnitId,
};
use runmat_hir::FunctionId;

use super::SelectedGraphCallableFacts;
use crate::bytecode::program::LexicalBindingKind;
use crate::FunctionRegistry;

/// Project the exact selected bodies used for executable dispatch. This is
/// also the Core admission seam for an immutable deferred program; neither a
/// display name nor a later published registry is consulted at spawn.
pub fn selected_graph_callable_facts(
    registry: &FunctionRegistry,
    unit: GraphProgramUnitId,
) -> Result<SelectedGraphCallableFacts, String> {
    selected_graph_callable_facts_in_owner(registry, unit, None, &[])
}

/// Project only bodies that retain one exact selected source issuance. A
/// bound static dependency graph may span several admitted products; its
/// registry remains whole for definition leases and lexical parent lookup,
/// while each body's graph owner points at its own ordered V3 unit.
pub fn selected_graph_callable_facts_for_owner(
    registry: &FunctionRegistry,
    unit: GraphProgramUnitId,
    owner: &runmat_runtime::source_context::SelectedProgramSourceOwner,
    selected_units: &[(
        GraphProgramUnitId,
        runmat_runtime::source_context::SelectedProgramSourceOwner,
    )],
) -> Result<SelectedGraphCallableFacts, String> {
    selected_graph_callable_facts_in_owner(registry, unit, Some(owner), selected_units)
}

fn selected_graph_callable_facts_in_owner(
    registry: &FunctionRegistry,
    unit: GraphProgramUnitId,
    selected_owner: Option<&runmat_runtime::source_context::SelectedProgramSourceOwner>,
    selected_units: &[(
        GraphProgramUnitId,
        runmat_runtime::source_context::SelectedProgramSourceOwner,
    )],
) -> Result<SelectedGraphCallableFacts, String> {
    let mut layouts = HashMap::new();
    let mut proofs = HashMap::new();
    let mut bound_bindings = HashMap::new();
    for body in registry.selected_execution_bodies() {
        let Some(origin) = body.binding_origin else {
            continue;
        };
        if selected_owner.is_some_and(|selected| {
            body.selected_source_owner
                .as_ref()
                .is_none_or(|actual| !actual.same_issuance(selected))
        }) {
            continue;
        }
        let selected = GraphBoundOwner { unit, origin };
        let binding = registry
            .exact_bound_binding(body.function)
            .ok_or("admitted graph callable has no exact definition lease")?;
        if bound_bindings.insert(selected, binding).is_some() {
            return Err("admitted registry repeats a portable function origin".into());
        }
        let owner = body
            .selected_source_owner
            .as_ref()
            .ok_or("admitted callable has no selected source owner")?;
        if !owner.is_live() {
            return Err("admitted callable source owner has retired".into());
        }
        let source = body
            .source_id
            .and_then(crate::BytecodeSourceRef::runtime)
            .ok_or("admitted callable has no issued source reference")?;
        let ordinal = owner
            .source_ordinal_for(source)
            .ok_or("admitted callable source differs from its selected owner")?;
        let record = owner
            .portable_map()
            .entries
            .get(usize::try_from(ordinal.0).map_err(|_| "source ordinal exceeds usize")?)
            .filter(|record| record.source_id == ordinal)
            .ok_or("admitted callable source ordinal is absent from its portable map")?;
        let issued = owner
            .selected_sources()
            .find_map(|(candidate, _, info)| (candidate == ordinal).then_some(info))
            .ok_or("admitted callable has no issued source record")?;
        let mut seen = HashSet::new();
        let mut bindings = body
            .lexical_binding_slots
            .iter()
            .map(|slot| {
                if !seen.insert(slot.binding) {
                    return Err("admitted function repeats a lexical binding");
                }
                let kind = match slot.kind {
                    LexicalBindingKind::Owned => GraphBindingKind::Owned,
                    LexicalBindingKind::Inherited => GraphBindingKind::Inherited,
                    LexicalBindingKind::Transit => GraphBindingKind::Transit,
                };
                Ok((slot.binding, kind))
            })
            .collect::<Result<Vec<_>, &'static str>>()?;
        bindings.sort_by_key(|(binding, _)| *binding);
        let activation_parent = nearest_activation_parent(
            body.lexical_parent,
            registry,
            unit,
            selected_owner,
            selected_units,
        )?;
        if layouts
            .insert(
                selected,
                GraphLexicalLayout {
                    activation_parent,
                    bindings,
                },
            )
            .is_some()
        {
            return Err("admitted registry repeats a portable function origin".into());
        }
        let captures = capture_proof(body, activation_parent);
        // A constructor handle is issued under the public class identity.
        // The admitted method body retains its internal, class-qualified
        // display name for code and declaration matching.
        let proof = runmat_runtime::execution::value_codec::GraphCallableProof {
            qualified_name: graph_callable_name(body),
            source_role: body.source_role,
            issued_source_file: issued
                .fullpath_name
                .as_ref()
                .map(|path| path.to_string())
                .or_else(|| body.source_file.clone()),
            source_file: Some(record.relative_path.clone()),
            captures,
        };
        proofs.insert(selected, proof);
    }
    Ok(SelectedGraphCallableFacts {
        layouts,
        proofs,
        bindings: bound_bindings,
    })
}

pub(super) fn graph_callable_name(body: &crate::FunctionBytecode) -> String {
    body.class_method_owner
        .as_ref()
        .filter(|owner| owner.role == runmat_types::ClassMethodRole::Constructor)
        .map(|owner| owner.declaring_class.display_name().to_owned())
        .unwrap_or_else(|| body.display_name.clone())
}

pub(super) fn capture_proof(
    body: &crate::FunctionBytecode,
    activation_parent: Option<GraphBoundOwner>,
) -> runmat_runtime::execution::value_codec::GraphCaptureProof {
    use runmat_runtime::execution::value_codec::{GraphCaptureProof, GraphNestedZeroCopy};
    if body.source_role == runmat_types::FunctionSourceRole::Nested {
        GraphCaptureProof::NestedShared {
            vm_copied: body.capture_slots.len(),
            zero_copy: if activation_parent.is_some() && body.captures_owned_without_copies(true) {
                GraphNestedZeroCopy::SharedActivation
            } else if body.captures_owned_without_copies(false) {
                GraphNestedZeroCopy::CanonicalStorage
            } else {
                GraphNestedZeroCopy::RequiresCopies
            },
        }
    } else if body.class_method_owner.as_ref().is_some_and(|owner| {
        owner.role == runmat_types::ClassMethodRole::Ordinary && !owner.is_static
    }) {
        GraphCaptureProof::InstanceMethodReceiver
    } else {
        GraphCaptureProof::Exact(body.capture_slots.len())
    }
}

pub(super) fn nearest_activation_parent(
    mut parent: Option<FunctionId>,
    registry: &FunctionRegistry,
    unit: GraphProgramUnitId,
    selected_owner: Option<&runmat_runtime::source_context::SelectedProgramSourceOwner>,
    selected_units: &[(
        GraphProgramUnitId,
        runmat_runtime::source_context::SelectedProgramSourceOwner,
    )],
) -> Result<Option<GraphBoundOwner>, String> {
    let mut seen = HashSet::new();
    while let Some(id) = parent {
        if !seen.insert(id) {
            return Err("admitted lexical parent chain is cyclic".into());
        }
        let body = registry
            .get(id)
            .ok_or("admitted lexical parent is absent from the function registry")?;
        if body.creates_lexical_environment {
            let parent_unit = if selected_owner.is_some() {
                let owner = body
                    .selected_source_owner
                    .as_ref()
                    .ok_or("admitted activation parent has no selected source owner")?;
                let matches = selected_units
                    .iter()
                    .filter(|(_, candidate)| candidate.same_issuance(owner))
                    .map(|(unit, _)| *unit)
                    .collect::<Vec<_>>();
                match matches.as_slice() {
                    [unit] => *unit,
                    [] => return Err("admitted activation parent has no selected V3 unit".into()),
                    _ => return Err("admitted activation parent has ambiguous V3 units".into()),
                }
            } else {
                unit
            };
            return body
                .binding_origin
                .map(|origin| {
                    Some(GraphBoundOwner {
                        unit: parent_unit,
                        origin,
                    })
                })
                .ok_or_else(|| "admitted activation parent has no portable origin".into());
        }
        parent = body.lexical_parent;
    }
    Ok(None)
}

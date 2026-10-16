use runmat_types::{
    AssignmentCreationPolicy, AssignmentShapePolicy, DynamicReason, IndexSelectorFact,
    LiteralValue, MutationContract, PlaceMutationKind, ValueFact, ValueKindFact,
};

use crate::{MirPlace, MirSequenceTarget, SequenceRootCreation};

use super::FlowState;

pub(super) fn assign_sequence_target(
    target: &MirSequenceTarget,
    output: Option<&ValueFact>,
    state: &mut FlowState,
) {
    let Some(local) = place_root_local(target.base()) else {
        return;
    };
    // A runtime-width comma list, nested path, or non-cell receiver must not
    // inherit an invented cell fact. Only direct scalar brace writes admit
    // the existing indexed-mutation inference.
    if let (
        MirSequenceTarget::CellContents {
            base: MirPlace::Local(base_local),
            indexing,
        },
        Some(payload),
    ) = (target, output)
    {
        let facts = state.value_facts();
        let selectors = crate::analysis::dataflow::index_selectors(indexing, &facts);
        let scalar_selection = !selectors.is_empty()
            && selectors
                .iter()
                .all(|selector| matches!(selector, IndexSelectorFact::KnownOneBasedIndex(index) if *index > 0));
        if scalar_selection {
            let prior = &state.locals[base_local.0];
            let base = if prior.assignment == crate::analysis::InitFact::Unassigned
                && target.root_creation() == SequenceRootCreation::EmptyCell
            {
                Some(runmat_types::infer_cell_aggregate(&[]).fact)
            } else {
                prior
                    .fact
                    .clone()
                    .filter(|fact| matches!(fact.kind, ValueKindFact::Cell(_)))
            };
            if let Some(base) = base {
                let inferred = runmat_types::infer_index_mutation(
                    &base,
                    &selectors,
                    payload,
                    MutationContract {
                        kind: PlaceMutationKind::CellAssign,
                        creation: AssignmentCreationPolicy::CreateArrayByIndex,
                        shape: AssignmentShapePolicy::MatlabCompatible,
                    },
                );
                if inferred.diagnostics.is_empty() {
                    state.locals[local.0].set(inferred.fact, LiteralValue::Unknown);
                    return;
                }
            }
        }
    }
    state.locals[local.0].set(
        ValueFact::unknown(DynamicReason::RuntimeValue),
        LiteralValue::Unknown,
    );
}

fn place_root_local(place: &MirPlace) -> Option<crate::MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Binding(_) => None,
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            place_root_local(base)
        }
    }
}

use runmat_types::{LiteralValue, ValueFact};

use crate::{MirPlace, MirRootStep, SequenceRootCreation};

use crate::analysis::engine::FlowState;

pub(crate) fn assign_place(
    place: &MirPlace,
    fact: ValueFact,
    literal: LiteralValue,
    index_bounds: Option<runmat_types::ScalarIndexBounds>,
    mutation: Option<&crate::MirPlaceMutation>,
    state: &mut FlowState,
) {
    if let MirPlace::Local(local) = place {
        state.locals[local.0].set(fact, literal);
        state.locals[local.0].index_bounds = index_bounds;
        return;
    }
    let mut facts = state.value_facts();
    // The runtime may create an empty Cell only when the first indexed root
    // step is brace indexing and the binding is proven absent. Pass that
    // lifecycle proof into the existing indexed-mutation inference instead
    // of treating a generic Unknown value as an absent cell.
    let root_creation = match place.first_root_step() {
        Some(MirRootStep::Index(runmat_hir::IndexKind::Brace))
            if mutation.is_some_and(|mutation| {
                mutation.kind != runmat_hir::PlaceMutationKind::Delete
                    && mutation.creation_policy
                        != runmat_hir::AssignmentCreationPolicy::ExistingOnly
            }) =>
        {
            SequenceRootCreation::EmptyCell
        }
        _ => SequenceRootCreation::Existing,
    };
    if let Some((local, empty_cell)) = state.absent_cell_root_for_place(place, root_creation) {
        facts[local.0] = Some(empty_cell);
    }
    crate::analysis::dataflow::assign_place_fact_with_bounds(
        place,
        fact,
        mutation,
        &mut facts,
        &|local| state.locals[local.0].index_bounds,
    );
    state.replace_value_facts(facts);
    if let Some(root) = place.root_local() {
        // Stores can change values while leaving every type/shape fact equal.
        state.locals[root.0].literal = LiteralValue::Unknown;
        state.locals[root.0].index_bounds = None;
    }
}

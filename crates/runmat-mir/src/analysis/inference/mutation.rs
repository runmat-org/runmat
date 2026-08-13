use runmat_types::{LiteralValue, ValueFact};

use crate::MirPlace;

use crate::analysis::engine::FlowState;

pub(crate) fn assign_place(
    place: &MirPlace,
    fact: ValueFact,
    literal: LiteralValue,
    mutation: Option<&crate::MirPlaceMutation>,
    state: &mut FlowState,
) {
    if let MirPlace::Local(local) = place {
        state.locals[local.0].set(fact, literal);
        return;
    }
    let mut facts = state.value_facts();
    crate::analysis::dataflow::assign_place_fact(place, fact, mutation, &mut facts);
    state.replace_value_facts(facts);
}

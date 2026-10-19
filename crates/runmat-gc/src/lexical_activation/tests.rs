use super::*;
use crate::{
    gc_allocate_lexical_activation_rooted, gc_collect_major, gc_test_context,
    gc_with_lexical_activation, gc_with_lexical_activation_mut, gc_with_value,
};
use runmat_value::{Closure, LexicalEnvironment};

#[test]
fn rooted_activation_traces_its_closure_and_unrooted_cycle_is_collectible() {
    gc_test_context(|| {
        let root = gc_allocate_lexical_activation_rooted(LexicalActivation::empty()).unwrap();
        let handle = root.handle();
        assert!(gc_with_value(&handle, |_| ()).is_err());
        assert!(gc_with_lexical_activation(&handle, LexicalActivation::is_empty).unwrap());

        let binding = BindingId(17);
        let table = PreparedLexicalTable::prepare(1, 0, None).unwrap();
        let reserved = gc_with_lexical_activation_mut(&handle, |activation| {
            activation.reserve_owned(binding, Some(table))
        })
        .unwrap();
        assert!(matches!(reserved, LexicalReservationOutcome::Applied(_)));

        let closure = Value::Closure(Closure::new(
            "nested".into(),
            None,
            runmat_types::FunctionSourceRole::Nested,
            None,
            Vec::new(),
            Some(LexicalEnvironment::new(handle, None)),
        ));
        let assigned = gc_with_lexical_activation_mut(&handle, |activation| {
            activation.assign(binding, LexicalValueOwner::unbounded(closure))
        })
        .unwrap();
        assert!(matches!(assigned, Ok(None)));

        gc_collect_major().unwrap();
        assert_eq!(
            gc_with_lexical_activation(&handle, |activation| activation.slot_state(binding))
                .unwrap(),
            Some(LexicalSlotState::Assigned)
        );
        drop(root);
        gc_collect_major().unwrap();
        assert!(gc_with_lexical_activation(&handle, |_| ()).is_err());
    });
}

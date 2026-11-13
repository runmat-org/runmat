use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn sequence_branch_constructors_follow_their_positional_contract() {
    assert_constructor_contract(&new::ENTRY, &inference::NEW);
}

#[test]
fn branch_lists_accept_text_scalars_string_arrays_and_text_cells() {
    use crate::fea_contract::study_types::test_support::{cell_of, infer, real};
    use runmat_types::{ShapeFact, StorageFact, ValueFact, ValueKindFact};
    let text = ValueFact::scalar(ValueKindFact::String);
    let strings = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    for consumers in [text.clone(), strings, cell_of(text.clone(), 3)] {
        let inferred = infer(
            &new::ENTRY,
            vec![text.clone(), text.clone(), text.clone(), consumers],
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
    }
    let numeric_cell = cell_of(real(ShapeFact::Scalar), 2);
    assert!(!infer(
        &new::ENTRY,
        vec![text.clone(), text.clone(), text, numeric_cell]
    )
    .diagnostics
    .is_empty());
}

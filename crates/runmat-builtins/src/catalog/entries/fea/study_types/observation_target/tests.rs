use super::*;
use crate::fea_contract::study_types::test_support::assert_constructor_contract;

#[test]
fn observation_target_constructors_follow_their_positional_contract() {
    for (entry, signature) in [
        (&boundary::ENTRY, &inference::BOUNDARY),
        (&volume::ENTRY, &inference::VOLUME),
        (&point::ENTRY, &inference::POINT),
        (&path::ENTRY, &inference::PATH),
        (&interface::ENTRY, &inference::INTERFACE),
        (&port::ENTRY, &inference::PORT),
        (&connector::ENTRY, &inference::CONNECTOR),
        (&constraint::ENTRY, &inference::CONSTRAINT),
        (&whole_model::ENTRY, &inference::WHOLE_MODEL),
    ] {
        assert_constructor_contract(entry, signature);
    }
}

#[test]
fn path_positions_require_three_columns_and_two_rows() {
    use crate::fea_contract::study_types::{
        inference::POSITIONS_CODE,
        test_support::{infer, real},
    };
    use runmat_types::ShapeFact;
    for shape in [vec![Some(3), Some(2)], vec![Some(1), Some(3)]] {
        let rejected = infer(&path::ENTRY, vec![real(ShapeFact::from(shape))]);
        assert!(rejected
            .diagnostics
            .iter()
            .any(|d| d.code == POSITIONS_CODE));
    }
    let open = infer(
        &path::ENTRY,
        vec![real(ShapeFact::from(vec![None, Some(3)]))],
    );
    assert!(open.diagnostics.is_empty());
}

#[test]
fn region_targets_reject_other_object_classes() {
    use crate::fea_contract::study_types::{
        classes::FEA_POSITION_CLASS,
        inference::OBJECT_CODE,
        test_support::{infer, object},
    };
    let rejected = infer(&boundary::ENTRY, vec![object(Some(FEA_POSITION_CLASS))]);
    assert!(rejected.diagnostics.iter().any(|d| d.code == OBJECT_CODE));
    assert!(infer(&volume::ENTRY, vec![object(None)])
        .diagnostics
        .is_empty());
}

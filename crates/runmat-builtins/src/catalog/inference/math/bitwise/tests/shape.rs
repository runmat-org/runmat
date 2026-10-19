use super::*;

#[test]
fn trailing_singletons_and_unknown_data_stay_sound() {
    let call = request_many(vec![
        numeric(NumericClass::UInt8, vec![Some(2), Some(1)]),
        numeric(NumericClass::Double, vec![Some(2), Some(1), Some(1)]),
    ]);
    let result = infer(
        crate::BitwiseInferenceRule::Get,
        &call,
        &crate::BITGET_CATALOG_ENTRY,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1), Some(1)])
    );
    let call = request(ValueFact::unknown(
        runmat_types::DynamicReason::RuntimeValue,
    ));
    let result = infer(
        crate::BitwiseInferenceRule::Complement,
        &call,
        &crate::BITCMP_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

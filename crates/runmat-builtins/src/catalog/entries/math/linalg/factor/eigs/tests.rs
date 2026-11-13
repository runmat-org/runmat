use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, EffectKind, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

#[test]
fn typed_options_preserve_known_geometry_and_drop_is_explicit() {
    let numeric = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::proven(
                numeric.clone(),
                ShapeFact::from(vec![Some(8), Some(8)]),
                StorageFact::Sparse,
            ),
            ValueFact::scalar(numeric),
        ],
        literals: LiteralContext {
            literal_args: vec![LiteralValue::Unknown, LiteralValue::Number(2.0)],
        },
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
    };
    let baseline = infer_catalog_call(&EIGS_CATALOG_ENTRY, &request);
    assert_eq!(
        baseline.outputs[0].shape,
        ShapeFact::from(vec![Some(8), Some(2)])
    );
    assert!(!baseline.effects.0.contains(&EffectKind::Unknown));
    assert!(!baseline.effects.0.contains(&EffectKind::HostCallback));
    for (name, value) in [
        ("Tolerance", LiteralValue::Number(1e-6)),
        ("FailureTreatment", LiteralValue::String("drop".into())),
    ] {
        request
            .arguments
            .push(ValueFact::scalar(ValueKindFact::String));
        request.arguments.push(ValueFact::unknown(
            runmat_types::DynamicReason::RuntimeValue,
        ));
        request
            .literals
            .literal_args
            .push(LiteralValue::String(name.into()));
        request.literals.literal_args.push(value);
        let result = infer_catalog_call(&EIGS_CATALOG_ENTRY, &request);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![
                Some(8),
                if EigsOption::from_name(name) == Some(EigsOption::Failure) {
                    None
                } else {
                    Some(2)
                }
            ])
        );
    }
}

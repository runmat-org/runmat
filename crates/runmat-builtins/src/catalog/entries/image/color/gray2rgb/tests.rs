use super::*;
use crate::BuiltinParamType;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericFact, OutputSelection, RequestedOutputCount,
};

fn request(kind: ValueKindFact, shape: ShapeFact) -> CallRequest {
    CallRequest {
        arguments: vec![ValueFact::proven(kind, shape, StorageFact::Dense)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn gray2rgb_preserves_exact_integer_and_logical_class_and_appends_channel() {
    assert_eq!(
        GRAY2RGB_DESCRIPTOR.signatures[0].outputs[0].ty,
        BuiltinParamType::Any
    );
    for kind in [
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Logical,
    ] {
        let inferred = infer(
            &request(kind.clone(), ShapeFact::from(vec![Some(2), Some(4)])),
            &GRAY2RGB_CATALOG_ENTRY,
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs[0].kind, kind);
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(4), Some(3)])
        );
        assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
    }
}

#[test]
fn gray2rgb_unknown_and_invalid_shapes_do_not_invent_dimensions() {
    let unknown = infer(
        &request(ValueKindFact::Unknown, ShapeFact::Unknown),
        &GRAY2RGB_CATALOG_ENTRY,
    );
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(
        unknown.outputs[0].certainty,
        runmat_types::CertaintyFact::Dynamic(runmat_types::DynamicReason::RuntimeValue)
    );
    let invalid = infer(
        &request(
            ValueKindFact::Logical,
            ShapeFact::from(vec![Some(1), Some(2), Some(3)]),
        ),
        &GRAY2RGB_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-GRAY2RGB-SHAPE"));
}

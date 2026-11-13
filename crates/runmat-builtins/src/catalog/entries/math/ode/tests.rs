use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(entry: &crate::BuiltinCatalogEntry, outputs: usize) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![
                ValueFact::unknown(DynamicReason::UnresolvedCallable),
                numeric(
                    NumericClass::Single,
                    ShapeFact::from(vec![Some(1), Some(4)]),
                ),
                numeric(
                    NumericClass::UInt16,
                    ShapeFact::from(vec![Some(2), Some(1)]),
                ),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn supported_solver_outputs_are_double_host_arrays_with_requested_geometry() {
    for entry in [
        &ODE15S_CATALOG_ENTRY,
        &ODE23_CATALOG_ENTRY,
        &ODE45_CATALOG_ENTRY,
    ] {
        let y_only = infer(entry, 1);
        assert_eq!(y_only.outputs.len(), 1);
        assert_eq!(
            y_only.outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(2)])
        );
        assert_eq!(
            y_only.outputs[0].numeric().unwrap().class,
            NumericClass::Double
        );
        let two = infer(entry, 2);
        assert_eq!(
            two.outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(1)])
        );
        assert_eq!(
            two.outputs[1].shape,
            ShapeFact::from(vec![Some(4), Some(2)])
        );
        assert!(infer(entry, 0).outputs.is_empty());
    }
}

#[test]
fn known_invalid_options_and_state_are_diagnosed_before_execution() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "RelTol".into(),
        numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(1), Some(2)]),
        ),
    );
    fields.insert("MaxSteps".into(), ValueFact::scalar(ValueKindFact::String));
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::UnresolvedCallable),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(1), Some(1)]),
            ),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(0), Some(1)]),
            ),
            ValueFact::proven(
                ValueKindFact::Struct(StructFact::scalar(fields, true)),
                ShapeFact::Scalar,
                StorageFact::Scalar,
            ),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = crate::infer_catalog_call(&ODE45_CATALOG_ENTRY, &request);
    assert!(result.diagnostics.len() >= 3);
    request.arguments[1] = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    assert!(
        crate::infer_catalog_call(&ODE45_CATALOG_ENTRY, &request)
            .diagnostics
            .len()
            >= 2
    );
}

#[test]
fn public_option_spellings_decode_to_closed_identities() {
    assert_eq!(
        OdeOptionField::from_public_name("rElToL"),
        Some(OdeOptionField::RelTol)
    );
    assert_eq!(
        OdeOptionField::from_public_name("MaxSteps"),
        Some(OdeOptionField::MaxSteps)
    );
    assert_eq!(OdeOptionField::from_public_name("Events"), None);
    assert_eq!(
        OdeOptionField::from_public_name("rEfInE"),
        Some(OdeOptionField::Refine)
    );
    assert!(!OdeOptionField::Refine.is_floating_control());
    assert_eq!(OdeOptionField::ALL.len(), 6);
}

#[test]
fn scalar_single_controls_and_empty_numeric_defaults_match_solver_admission() {
    let fields = BTreeMap::from([
        (
            "RelTol".into(),
            numeric(NumericClass::Single, ShapeFact::Scalar),
        ),
        (
            "Refine".into(),
            numeric(NumericClass::Double, ShapeFact::Scalar),
        ),
        (
            "AbsTol".into(),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(0), Some(0)]),
            ),
        ),
    ]);
    let request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::UnresolvedCallable),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(1), Some(2)]),
            ),
            numeric(NumericClass::Double, ShapeFact::Scalar),
            ValueFact::proven(
                ValueKindFact::Struct(StructFact::scalar(fields, true)),
                ShapeFact::Scalar,
                StorageFact::Scalar,
            ),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    assert!(crate::infer_catalog_call(&ODE45_CATALOG_ENTRY, &request)
        .diagnostics
        .is_empty());
}

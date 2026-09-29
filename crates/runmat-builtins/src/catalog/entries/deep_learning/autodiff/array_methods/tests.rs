use super::*;
use runmat_types::{
    standard, LiteralContext, NumericClass, NumericDomain, NumericFact, ObjectFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
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

fn dlarray(data: ValueFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::DL_ARRAY.owned()),
            properties: BTreeMap::from([(crate::DLARRAY_DATA_PROPERTY.into(), data.clone())]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        data.shape,
        StorageFact::Opaque,
    )
}

fn infer(
    entry: &crate::BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        entry,
        &runmat_types::CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

fn output_data(inference: &runmat_types::CallInference) -> &ValueFact {
    let ValueKindFact::Object(object) = &inference.outputs[0].kind else {
        panic!("dlarray output")
    };
    assert_eq!(object.runtime_class, Some(standard::DL_ARRAY.owned()));
    &object.properties[crate::DLARRAY_DATA_PROPERTY]
}

#[test]
fn binary_methods_infer_symmetric_single_and_known_geometry() {
    let single = dlarray(numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(1), Some(3)]),
    ));
    let scalar = numeric(NumericClass::Double, ShapeFact::Scalar);
    for entry in [
        &DLARRAY_PLUS_CATALOG_ENTRY,
        &DLARRAY_MINUS_CATALOG_ENTRY,
        &DLARRAY_TIMES_CATALOG_ENTRY,
        &DLARRAY_RDIVIDE_CATALOG_ENTRY,
    ] {
        for operands in [
            vec![single.clone(), scalar.clone()],
            vec![scalar.clone(), single.clone()],
        ] {
            let result = infer(entry, operands);
            assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
            assert_eq!(
                output_data(&result).numeric().unwrap().class,
                NumericClass::Single
            );
            assert_eq!(
                output_data(&result).shape,
                ShapeFact::from(vec![Some(1), Some(3)])
            );
        }
    }
}

#[test]
fn matrix_and_reduction_facts_keep_single_without_overclaiming_unknowns() {
    let single = dlarray(numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(3)]),
    ));
    let right = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(3), Some(4)]),
    );
    let result = infer(&DLARRAY_MTIMES_CATALOG_ENTRY, vec![single.clone(), right]);
    assert_eq!(
        output_data(&result).numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        output_data(&result).shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );
    let reduced = infer(&DLARRAY_SUM_CATALOG_ENTRY, vec![single]);
    assert_eq!(
        output_data(&reduced).numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(output_data(&reduced).shape, ShapeFact::Scalar);
    let unknown = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    let uncertain = infer(
        &DLARRAY_PLUS_CATALOG_ENTRY,
        vec![unknown, numeric(NumericClass::Double, ShapeFact::Scalar)],
    );
    assert!(output_data(&uncertain).numeric().is_none());
}

#[test]
fn known_integer_data_rejected_statically() {
    let integer = dlarray(numeric(NumericClass::UInt64, ShapeFact::Scalar));
    let result = infer(
        &DLARRAY_PLUS_CATALOG_ENTRY,
        vec![integer, numeric(NumericClass::Double, ShapeFact::Scalar)],
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-ARRAY-CLASS"));
}

#[test]
fn logical_inputs_convert_to_double_at_arithmetic_boundary() {
    let logical = dlarray(ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    ));
    let scalar = numeric(NumericClass::Double, ShapeFact::Scalar);
    let added = infer(&DLARRAY_PLUS_CATALOG_ENTRY, vec![logical.clone(), scalar]);
    assert_eq!(
        output_data(&added).numeric().unwrap().class,
        NumericClass::Double
    );
    let summed = infer(&DLARRAY_SUM_CATALOG_ENTRY, vec![logical]);
    assert_eq!(
        output_data(&summed).numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(output_data(&summed).shape, ShapeFact::Scalar);
}

#[test]
fn arithmetic_declares_ambient_tape_mutation() {
    for entry in [
        &DLARRAY_PLUS_CATALOG_ENTRY,
        &DLARRAY_MINUS_CATALOG_ENTRY,
        &DLARRAY_TIMES_CATALOG_ENTRY,
        &DLARRAY_RDIVIDE_CATALOG_ENTRY,
        &DLARRAY_MTIMES_CATALOG_ENTRY,
        &DLARRAY_SUM_CATALOG_ENTRY,
    ] {
        assert_eq!(entry.contract.purity, crate::BuiltinPurity::Impure);
        assert!(entry
            .contract
            .effects
            .contains(&runmat_types::EffectKind::EnvironmentWrite));
    }
}

#[test]
fn sum_diagnoses_known_unsupported_selector() {
    let input = dlarray(numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(2)]),
    ));
    let request = runmat_types::CallRequest {
        arguments: vec![input, ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::new(vec![
            runmat_types::LiteralValue::Unknown,
            runmat_types::LiteralValue::String("rows".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = crate::infer_catalog_call(&DLARRAY_SUM_CATALOG_ENTRY, &request);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-SUM-SELECTOR"));
}

#[test]
fn known_incompatible_shapes_diagnose_but_unknown_dimensions_remain_dynamic() {
    let left = dlarray(numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(1), Some(2)]),
    ));
    let right = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(1)]),
    );
    let elementwise = infer(&DLARRAY_PLUS_CATALOG_ENTRY, vec![left, right]);
    assert!(elementwise
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-ARRAY-SHAPE"));

    let left = dlarray(numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(3)]),
    ));
    let right = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(4), Some(2)]),
    );
    let matrix = infer(&DLARRAY_MTIMES_CATALOG_ENTRY, vec![left, right]);
    assert!(matrix
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-ARRAY-SHAPE"));

    let left = dlarray(numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), None]),
    ));
    let right = numeric(NumericClass::Double, ShapeFact::from(vec![None, Some(4)]));
    let dynamic = infer(&DLARRAY_MTIMES_CATALOG_ENTRY, vec![left, right]);
    assert!(!dynamic
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-ARRAY-SHAPE"));
    assert_eq!(
        output_data(&dynamic).shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );

    let scalar = numeric(NumericClass::Double, ShapeFact::Scalar);
    let array = dlarray(numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(3)]),
    ));
    let scaled = infer(&DLARRAY_MTIMES_CATALOG_ENTRY, vec![scalar, array]);
    assert!(!scaled
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-ARRAY-SHAPE"));
    assert_eq!(
        output_data(&scaled).shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
}

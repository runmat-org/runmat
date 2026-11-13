use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        StorageFact::Dense,
    )
}

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn complex_values_and_matrix_geometry_have_sound_output_facts() {
    let inferred = inference::infer(
        &request(vec![
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                ShapeFact::from(vec![Some(3), Some(1)]),
            ),
            numeric(
                NumericClass::Single,
                NumericDomain::Complex,
                ShapeFact::from(vec![Some(3), Some(2)]),
            ),
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                ShapeFact::from(vec![Some(1), Some(4)]),
            ),
        ]),
        &INTERP1Q_CATALOG_ENTRY,
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(2)])
    );
    assert_eq!(
        inferred.outputs[0].numeric().unwrap(),
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }
    );
}

#[test]
fn vector_values_preserve_query_shape_and_real_domain() {
    let inferred = inference::infer(
        &request(vec![
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                ShapeFact::from(vec![Some(3), Some(1)]),
            ),
            numeric(
                NumericClass::Single,
                NumericDomain::Real,
                ShapeFact::from(vec![Some(3), Some(1)]),
            ),
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                ShapeFact::from(vec![Some(2), Some(2)]),
            ),
        ]),
        &INTERP1Q_CATALOG_ENTRY,
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn incomplete_and_wrong_arity_inputs_do_not_invent_output_geometry() {
    let inferred = inference::infer(&request(vec![]), &INTERP1Q_CATALOG_ENTRY);
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-INTERP1Q-ARITY"));
}

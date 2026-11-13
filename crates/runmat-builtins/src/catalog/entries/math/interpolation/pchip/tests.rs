use super::*;
use runmat_types::*;

fn double(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn request(args: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments: args,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn constructor_has_scalar_pp_shape_and_public_fields() {
    let inferred = inference::infer(
        &request(vec![double(ShapeFact::from(vec![Some(1), Some(3)])); 2]),
        &PCHIP_CATALOG_ENTRY,
    );
    let ValueKindFact::Struct(structure) = &inferred.outputs[0].kind else {
        panic!("pchip constructor must yield a pp structure");
    };
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
    for field in ["form", "breaks", "coefs", "pieces", "order", "dim"] {
        assert!(structure.fields.contains_key(field));
    }
}

#[test]
fn vector_samples_preserve_query_shape_but_array_samples_do_not_invent_it() {
    let x = double(ShapeFact::from(vec![Some(1), Some(3)]));
    let query = double(ShapeFact::from(vec![Some(2), Some(2)]));
    let vector = inference::infer(
        &request(vec![
            x.clone(),
            double(ShapeFact::from(vec![Some(1), Some(3)])),
            query.clone(),
        ]),
        &PCHIP_CATALOG_ENTRY,
    );
    assert_eq!(vector.outputs[0].shape, query.shape);
    assert_eq!(
        vector.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    let array = inference::infer(
        &request(vec![
            x,
            double(ShapeFact::from(vec![Some(2), Some(3), Some(3)])),
            query,
        ]),
        &PCHIP_CATALOG_ENTRY,
    );
    assert_eq!(array.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn missing_inputs_and_complex_samples_are_diagnosed_without_panicking() {
    let missing = inference::infer(&request(vec![]), &PCHIP_CATALOG_ENTRY);
    assert!(missing
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PCHIP-INPUT"));
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let result = inference::infer(
        &request(vec![
            double(ShapeFact::from(vec![Some(1), Some(3)])),
            complex,
        ]),
        &PCHIP_CATALOG_ENTRY,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PCHIP-INPUT"));
}

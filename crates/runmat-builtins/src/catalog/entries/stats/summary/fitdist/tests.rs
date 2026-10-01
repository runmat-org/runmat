use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn fitdist_proves_typed_fitted_value_and_validates_family_and_frequency() {
    let sample = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(1)]),
        StorageFact::Dense,
    );
    let family = ValueFact::scalar(ValueKindFact::String);
    let good = request(
        vec![sample.clone(), family.clone()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("Normal".into()),
        ],
    );
    let inferred = infer(&good, &FITDIST_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("expected fitted object");
    };
    assert!(object
        .runtime_class
        .as_ref()
        .is_some_and(|class| class.is(runmat_types::standard::PROBABILITY_DISTRIBUTION)));
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);

    let invalid = request(
        vec![sample.clone(), family.clone()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("Unsupported".into()),
        ],
    );
    assert!(infer(&invalid, &FITDIST_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FITDIST-FAMILY"));
    let wrong_weights = request(
        vec![
            sample,
            family,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(2), Some(1)]),
                StorageFact::Dense,
            ),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("Normal".into()),
            LiteralValue::Character("Frequency".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(infer(&wrong_weights, &FITDIST_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FITDIST-FREQUENCY"));
}

#[test]
fn fitdist_logical_data_and_frequency_follow_runtime_admission() {
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(3), Some(1)]),
        StorageFact::Dense,
    );
    let request = request(
        vec![
            logical.clone(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            logical,
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("Normal".into()),
            LiteralValue::Character("Frequency".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(infer(&request, &FITDIST_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn fitdist_rejects_duplicate_or_nonvector_or_complex_frequency() {
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), Some(1)]),
        StorageFact::Dense,
    );
    let name = ValueFact::scalar(ValueKindFact::String);
    let vector = data.clone();
    let duplicate = request(
        vec![
            data.clone(),
            name.clone(),
            name.clone(),
            vector.clone(),
            name.clone(),
            vector,
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("Normal".into()),
            LiteralValue::Character("Frequency".into()),
            LiteralValue::Unknown,
            LiteralValue::Character("Frequency".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(infer(&duplicate, &FITDIST_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FITDIST-OPTION"));

    let matrix = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    let nonvector = request(
        vec![data.clone(), name.clone(), name.clone(), matrix],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("Normal".into()),
            LiteralValue::Character("Frequency".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(infer(&nonvector, &FITDIST_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FITDIST-FREQUENCY"));

    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(4), Some(1)]),
        StorageFact::Dense,
    );
    let complex_frequency = request(
        vec![data, name.clone(), name, complex],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("Normal".into()),
            LiteralValue::Character("Frequency".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(infer(&complex_frequency, &FITDIST_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FITDIST-FREQUENCY"));
}

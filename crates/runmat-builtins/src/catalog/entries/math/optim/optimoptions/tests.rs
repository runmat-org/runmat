use super::*;
use runmat_types::*;

#[test]
fn options_count_fact_retains_native_class_but_unknown_solver_stays_partial() {
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt64,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(1), Some(1)]),
                StorageFact::Dense,
            ),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("fminunc".into()),
            LiteralValue::String("MaxIter".into()),
            LiteralValue::Integer {
                text: "9007199254740993".into(),
                class: NumericClass::UInt64,
            },
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &OPTIMOPTIONS_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
        panic!("options");
    };
    assert!(fields.fields_complete);
    assert_eq!(
        fields.fields["MaxIter"].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(fields.fields["MaxIter"].storage, StorageFact::Scalar);
    assert!(fields.fields.contains_key("OptimalityTolerance"));
    assert!(fields.fields.contains_key("FunctionTolerance"));
    assert!(!fields.fields.contains_key("TolFun"));
    assert_eq!(
        fields.fields["MaxFunEvals"].shape,
        ShapeFact::Unknown,
        "the effective algorithm can select a dimension-dependent default marker"
    );
    let known = request.clone();
    request.literals.literal_args[1] = LiteralValue::Unknown;
    let result = inference::infer(&request, &OPTIMOPTIONS_CATALOG_ENTRY);
    let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
        panic!("options");
    };
    assert!(!fields.fields_complete);
    assert!(fields
        .fields
        .get("MaxIter")
        .and_then(ValueFact::numeric)
        .is_none());
    request
        .arguments
        .extend([known.arguments[1].clone(), known.arguments[2].clone()]);
    request.literals.literal_args.extend([
        known.literals.literal_args[1].clone(),
        known.literals.literal_args[2].clone(),
    ]);
    let result = inference::infer(&request, &OPTIMOPTIONS_CATALOG_ENTRY);
    let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
        panic!("options");
    };
    assert_eq!(
        fields.fields["MaxIter"].numeric().unwrap().class,
        NumericClass::UInt64
    );
    for complete in [false, true] {
        request = known.clone();
        let mut incoming = std::collections::BTreeMap::new();
        incoming.insert(
            "MaxIter".into(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        );
        incoming.insert("Solver".into(), ValueFact::scalar(ValueKindFact::String));
        request
            .arguments
            .push(ValueFact::scalar(ValueKindFact::Struct(
                StructFact::scalar(incoming, complete),
            )));
        let result = inference::infer(&request, &OPTIMOPTIONS_CATALOG_ENTRY);
        let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
            panic!("options");
        };
        assert!(
            fields
                .fields
                .get("MaxIter")
                .and_then(ValueFact::numeric)
                .is_none(),
            "unknown aliases or default skipping must not assert Double"
        );
        request
            .arguments
            .extend([known.arguments[1].clone(), known.arguments[2].clone()]);
        request.literals.literal_args.push(LiteralValue::Unknown);
        request.literals.literal_args.extend([
            known.literals.literal_args[1].clone(),
            known.literals.literal_args[2].clone(),
        ]);
        let result = inference::infer(&request, &OPTIMOPTIONS_CATALOG_ENTRY);
        let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
            panic!("options");
        };
        assert_eq!(
            fields.fields["MaxIter"].numeric().unwrap().class,
            NumericClass::UInt64
        );
    }
    request.arguments = vec![ValueFact::unknown(DynamicReason::RuntimeValue)];
    request.literals = LiteralContext::default();
    let result = inference::infer(&request, &OPTIMOPTIONS_CATALOG_ENTRY);
    let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
        panic!("options");
    };
    assert!(!fields.fields_complete);
    assert!(!fields.fields.contains_key("Solver"));
}

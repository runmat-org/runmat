use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn number(rows: usize, columns: usize, domain: NumericDomain) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn model(domain: NumericDomain) -> ValueFact {
    let mut properties = BTreeMap::new();
    for name in ["Numerator", "Denominator"] {
        properties.insert(name.into(), number(1, 2, domain));
    }
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn call(
    entry: &BuiltinCatalogEntry,
    args: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> CallInference {
    crate::infer_catalog_call(
        entry,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

#[test]
fn all_twelve_are_method_only_with_distinct_contracts_and_examples() {
    assert_eq!(ENTRIES.len(), 12);
    for entry in ENTRIES {
        assert_eq!(
            entry.descriptor.completion_policy,
            BuiltinCompletionPolicy::MethodOnly
        );
        assert_eq!(entry.descriptor.signatures.len(), 1);
        assert_eq!(entry.documentation.examples.len(), 1);
        assert!(matches!(
            entry.documentation.examples[0].verification,
            BuiltinExampleVerification::Assertions { .. }
        ));
        assert_eq!(
            entry.placement.residency,
            BuiltinResidencyPolicy::GatherToHost
        );
        assert_eq!(
            entry.placement.portability,
            BuiltinPortability::NativeAndWasm
        );
        assert!(!entry
            .descriptor
            .errors
            .iter()
            .any(|error| error.identifier == Some("RunMat:tf:IntegerPrecisionLoss")));
    }
}

#[test]
fn power_capacity_error_is_canonical_for_both_power_spellings() {
    for entry in [&TF_POWER_CATALOG_ENTRY, &TF_MPOWER_CATALOG_ENTRY] {
        assert!(entry.descriptor.errors.iter().any(|error| {
            error.code == TF_METHOD_ERROR_COEFFICIENT_CAPACITY.code
                && error.identifier == TF_METHOD_ERROR_COEFFICIENT_CAPACITY.identifier
        }));
    }
}

#[test]
fn each_tf_method_advertises_only_its_reachable_diagnostic_family() {
    const COMMON: &[&str] = &[
        "RunMat:tf:InvalidArgument",
        "RunMat:tf:InvalidModel",
        "RunMat:tf:UnsupportedModel",
        "RunMat:tf:InvalidCoefficients",
        "RunMat:tf:InvalidSampleTime",
        "RunMat:tf:InvalidVariable",
        "RunMat:tf:Internal",
    ];
    const BINARY: &[&str] = &[
        "RunMat:tf:DenominatorInvalid",
        "RunMat:tf:SampleTimeMismatch",
        "RunMat:tf:VariableMismatch",
        "RunMat:tf:UnsupportedDelay",
    ];
    const DIVISION: &[&str] = &["RunMat:tf:DivideByZero"];
    const POWER: &[&str] = &[
        "RunMat:tf:DenominatorInvalid",
        "RunMat:tf:UnsupportedDelay",
        "RunMat:tf:DivideByZero",
        "RunMat:tf:InvalidExponent",
        "RunMat:tf:CoefficientCapacity",
    ];

    for entry in ENTRIES {
        let mut expected = COMMON.to_vec();
        match entry.identity.name {
            "tf.uplus" | "tf.uminus" => {}
            "tf.plus" | "tf.minus" | "tf.times" | "tf.mtimes" => {
                expected.extend_from_slice(BINARY);
            }
            "tf.rdivide" | "tf.mrdivide" | "tf.ldivide" | "tf.mldivide" => {
                expected.extend_from_slice(BINARY);
                expected.extend_from_slice(DIVISION);
            }
            "tf.power" | "tf.mpower" => expected.extend_from_slice(POWER),
            other => panic!("unclassified tf method {other}"),
        }
        expected.sort_unstable();
        let mut actual = entry
            .descriptor
            .errors
            .iter()
            .map(|error| error.identifier.expect("stable tf identifier"))
            .collect::<Vec<_>>();
        actual.sort_unstable();
        assert_eq!(actual, expected, "{}", entry.identity.name);
    }
}

#[test]
fn real_model_binary_and_unary_results_have_host_tf_property_schema() {
    for entry in [
        &TF_PLUS_CATALOG_ENTRY,
        &TF_MINUS_CATALOG_ENTRY,
        &TF_TIMES_CATALOG_ENTRY,
        &TF_MTIMES_CATALOG_ENTRY,
        &TF_RDIVIDE_CATALOG_ENTRY,
        &TF_MRDIVIDE_CATALOG_ENTRY,
        &TF_LDIVIDE_CATALOG_ENTRY,
        &TF_MLDIVIDE_CATALOG_ENTRY,
    ] {
        let result = call(
            entry,
            vec![
                model(NumericDomain::Real),
                number(1, 1, NumericDomain::Real),
            ],
            vec![],
        );
        assert!(
            result.diagnostics.is_empty(),
            "{}: {:?}",
            entry.identity.name,
            result.diagnostics
        );
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        let ValueKindFact::Object(object) = &result.outputs[0].kind else {
            panic!("tf result")
        };
        assert_eq!(
            object.runtime_class,
            Some(standard::TRANSFER_FUNCTION.owned())
        );
        assert_eq!(
            object.properties["Numerator"].shape,
            ShapeFact::from(vec![Some(1), None])
        );
        assert_eq!(
            object.properties["Numerator"].numeric().unwrap().domain,
            NumericDomain::Real
        );
    }
    for entry in [&TF_UPLUS_CATALOG_ENTRY, &TF_UMINUS_CATALOG_ENTRY] {
        let result = call(entry, vec![model(NumericDomain::Real)], vec![]);
        assert!(result.diagnostics.is_empty());
        assert!(matches!(&result.outputs[0].kind, ValueKindFact::Object(_)));
    }
}

#[test]
fn known_invalid_scalar_shape_class_and_exponent_are_diagnosed() {
    let matrix = call(
        &TF_PLUS_CATALOG_ENTRY,
        vec![
            model(NumericDomain::Real),
            number(2, 2, NumericDomain::Real),
        ],
        vec![],
    );
    assert!(matrix
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TF-METHOD-SCALAR"));
    let unary_gain = call(
        &TF_UPLUS_CATALOG_ENTRY,
        vec![number(1, 1, NumericDomain::Real)],
        vec![],
    );
    assert!(unary_gain
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TF-METHOD-MODEL"));
    let two_gains = call(
        &TF_MTIMES_CATALOG_ENTRY,
        vec![
            number(1, 1, NumericDomain::Real),
            number(1, 1, NumericDomain::Real),
        ],
        vec![],
    );
    assert!(two_gains
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TF-METHOD-MODEL"));
    for entry in [&TF_POWER_CATALOG_ENTRY, &TF_MPOWER_CATALOG_ENTRY] {
        let result = call(
            entry,
            vec![
                model(NumericDomain::Real),
                number(1, 1, NumericDomain::Real),
            ],
            vec![LiteralValue::Unknown, LiteralValue::Number(1.5)],
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-TF-METHOD-EXPONENT"));
    }
}

#[test]
fn unknown_operands_keep_diagnostics_and_coefficient_domain_conservative() {
    let result = call(
        &TF_PLUS_CATALOG_ENTRY,
        vec![
            model(NumericDomain::Complex),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        vec![],
    );
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("tf result")
    };
    assert!(object.properties["Numerator"].numeric().is_none());
}

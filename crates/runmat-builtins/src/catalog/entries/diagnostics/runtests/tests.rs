use super::*;
use runmat_types::*;

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, count: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &RUNTESTS_CATALOG_ENTRY,
    )
}

#[test]
fn result_class_and_output_count_are_owned_by_identity() {
    let result = infer(vec![], vec![], 1);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("TestResult fact")
    };
    assert_eq!(
        object.runtime_class,
        Some(StaticClassIdentity::new("matlab.unittest.TestResult").owned())
    );
    assert!(!infer(vec![], vec![], 2).diagnostics.is_empty());
}

#[test]
fn integer_flags_and_parallel_literals_are_checked_without_target_access() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let invalid = infer(
        vec![text.clone(), integer],
        vec![
            LiteralValue::String("IncludeSubfolders".into()),
            LiteralValue::Integer {
                text: "2".into(),
                class: NumericClass::UInt64,
            },
        ],
        1,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-RUNTESTS-FLAG"));
    let parallel = infer(
        vec![text, ValueFact::scalar(ValueKindFact::Logical)],
        vec![
            LiteralValue::String("UseParallel".into()),
            LiteralValue::Bool(true),
        ],
        1,
    );
    assert!(parallel
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-RUNTESTS-PARALLEL"));
}

#[test]
fn target_numbers_and_resident_controls_do_not_infer_valid_calls() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(infer(vec![numeric], vec![LiteralValue::Number(1.0)], 1)
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-RUNTESTS-TARGET"));
    let mut resident = ValueFact::scalar(ValueKindFact::String);
    resident.residency = ResidencyFact::Device { provider: None };
    assert!(!infer(
        vec![resident],
        vec![LiteralValue::String("testSmoke.m".into())],
        1
    )
    .diagnostics
    .is_empty());
}

#[test]
fn discovery_contract_stays_an_incomplete_host_boundary() {
    assert_eq!(
        RUNTESTS_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    );
    assert_eq!(
        RUNTESTS_CATALOG_ENTRY.placement.preparation,
        BuiltinOperandPreparation::RuntimeOwned
    );
    assert_eq!(
        RUNTESTS_CATALOG_ENTRY.placement.fusion,
        BuiltinFusionPolicy::Boundary
    );
    assert_eq!(
        RuntestsOption::parse("Include_Subfolders"),
        Some(RuntestsOption::IncludeSubfolders)
    );
    assert_eq!(RuntestsOption::parse("Strict"), None);
}

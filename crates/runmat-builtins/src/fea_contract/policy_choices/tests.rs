use super::inference::{ARGUMENTS, CALLABLE, OPTION};
use crate::fea_contract::policy_types::classes::{
    FORCE_SIGNAL, PRESSURE_SIGNAL, SIGNAL_DOMAIN, SIGNAL_EXTRAPOLATION, SIGNAL_INTERPOLATION,
    SIGNAL_SAMPLING,
};
use crate::fea_contract::policy_types::conformance::{object, run};
use crate::fea_contract::policy_types::inference::{ARITY, NUMERIC, RESIDENT, SHAPE, TEXT, TYPED};
use crate::BuiltinInferenceRule;
use runmat_types::{
    CallRequest, CallableFact, CapabilitySet, CellFact, DynamicReason, LiteralContext,
    LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

mod pressure_samples {
    use crate::fea_contract::policy_choices::entry::time_samples_entry;
    time_samples_entry!(
        FIXTURE_CATALOG_ENTRY,
        name: "fea.PressureSignal.fromTimeSamplesFixture",
        class: crate::fea_contract::policy_types::classes::PRESSURE_SIGNAL,
        shape: Scalar,
        signature: "signal = fea.PressureSignal.fromTimeSamplesFixture(id, times, values, interpolation, extrapolation)",
        documentation: crate::BuiltinDocumentation::EMPTY,
    );
}

mod force_samples {
    use crate::fea_contract::policy_choices::entry::time_samples_entry;
    time_samples_entry!(
        FIXTURE_CATALOG_ENTRY,
        name: "fea.ForceSignal.fromTimeSamplesFixture",
        class: crate::fea_contract::policy_types::classes::FORCE_SIGNAL,
        shape: Vector3,
        signature: "signal = fea.ForceSignal.fromTimeSamplesFixture(id, times, values, interpolation, extrapolation)",
        documentation: crate::BuiltinDocumentation::EMPTY,
    );
}

mod force_function {
    use crate::fea_contract::policy_choices::entry::function_signal_entry;
    function_signal_entry!(
        FIXTURE_CATALOG_ENTRY,
        name: "fea.ForceSignal.fromFunctionFixture",
        class: crate::fea_contract::policy_types::classes::FORCE_SIGNAL,
        signature: "signal = fea.ForceSignal.fromFunctionFixture(id, callable, domain, sampling, Name, Value)",
        documentation: crate::BuiltinDocumentation::EMPTY,
    );
}

pub(crate) fn double(dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

pub(crate) fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

pub(crate) fn callable() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: false,
    }))
}

pub(crate) fn cell() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    )
}

pub(crate) fn codes(inference: &runmat_types::CallInference) -> Vec<(String, Option<usize>)> {
    inference
        .diagnostics
        .iter()
        .map(|diagnostic| (diagnostic.code.clone(), diagnostic.argument))
        .collect()
}

fn run_with_literals(
    entry: &crate::BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> runmat_types::CallInference {
    let BuiltinInferenceRule::Identity(rule) = entry.contract.inference_rule else {
        panic!("identity rule")
    };
    rule.infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        entry,
    )
}

fn result_class(inference: &runmat_types::CallInference) -> String {
    let ValueKindFact::Object(object) = &inference.outputs[0].kind else {
        panic!("object result")
    };
    object
        .runtime_class
        .as_ref()
        .unwrap()
        .display_name()
        .to_owned()
}

#[test]
fn scalar_time_samples_check_counts_policies_and_residency() {
    let entry = &pressure_samples::FIXTURE_CATALOG_ENTRY;
    let valid = vec![
        text(),
        double(&[1, 3]),
        double(&[3, 1]),
        object(SIGNAL_INTERPOLATION),
        object(SIGNAL_EXTRAPOLATION),
    ];
    let accepted = run(entry, valid.clone());
    assert!(
        accepted.diagnostics.is_empty(),
        "{:?}",
        accepted.diagnostics
    );
    assert_eq!(result_class(&accepted), PRESSURE_SIGNAL.display_name());
    assert_eq!(accepted.outputs[0].residency, ResidencyFact::Host);

    let mut mismatch = valid.clone();
    mismatch[2] = double(&[1, 4]);
    assert!(codes(&run(entry, mismatch)).contains(&(SHAPE.into(), Some(2))));

    let mut single = valid.clone();
    single[1] = double(&[1, 1]);
    single[2] = double(&[1, 1]);
    assert!(codes(&run(entry, single)).contains(&(SHAPE.into(), Some(1))));

    let mut swapped = valid.clone();
    swapped.swap(3, 4);
    let swapped = codes(&run(entry, swapped));
    assert!(
        swapped.contains(&(TYPED.into(), Some(3))) && swapped.contains(&(TYPED.into(), Some(4)))
    );

    let mut wrong_kinds = valid.clone();
    wrong_kinds[0] = double(&[1, 1]);
    wrong_kinds[1] = text();
    let wrong_kinds = codes(&run(entry, wrong_kinds));
    assert!(
        wrong_kinds.contains(&(TEXT.into(), Some(0)))
            && wrong_kinds.contains(&(NUMERIC.into(), Some(1)))
    );

    let mut resident = valid.clone();
    resident[2].residency = ResidencyFact::Device { provider: None };
    assert!(codes(&run(entry, resident)).contains(&(RESIDENT.into(), Some(2))));

    assert!(codes(&run(entry, valid[..4].to_vec()))
        .iter()
        .any(|(code, _)| code == ARITY));
    let unknown = vec![ValueFact::unknown(DynamicReason::RuntimeValue); 5];
    assert!(run(entry, unknown).diagnostics.is_empty());
}

#[test]
fn vector_time_samples_accept_either_orientation_of_three_components() {
    let entry = &force_samples::FIXTURE_CATALOG_ENTRY;
    let with_values = |values: ValueFact| {
        run(
            entry,
            vec![
                text(),
                double(&[1, 4]),
                values,
                object(SIGNAL_INTERPOLATION),
                object(SIGNAL_EXTRAPOLATION),
            ],
        )
    };
    assert!(with_values(double(&[4, 3])).diagnostics.is_empty());
    assert!(with_values(double(&[3, 4])).diagnostics.is_empty());
    assert!(codes(&with_values(double(&[4, 2]))).contains(&(SHAPE.into(), Some(2))));
    assert!(codes(&with_values(double(&[5, 3]))).contains(&(SHAPE.into(), Some(2))));
    assert_eq!(
        result_class(&with_values(double(&[4, 3]))),
        FORCE_SIGNAL.display_name()
    );
}

#[test]
fn function_signals_check_callable_policies_and_name_value_options() {
    let entry = &force_function::FIXTURE_CATALOG_ENTRY;
    let base = vec![
        text(),
        callable(),
        object(SIGNAL_DOMAIN),
        object(SIGNAL_SAMPLING),
    ];
    let accepted = run(entry, base.clone());
    assert!(
        accepted.diagnostics.is_empty(),
        "{:?}",
        accepted.diagnostics
    );
    assert_eq!(result_class(&accepted), FORCE_SIGNAL.display_name());

    let mut options = base.clone();
    options.extend([text(), cell(), text(), object(SIGNAL_EXTRAPOLATION)]);
    let mut literals = vec![LiteralValue::Unknown; 4];
    literals.extend([
        LiteralValue::String("Arguments".into()),
        LiteralValue::Unknown,
        LiteralValue::String("extrapolation".into()),
        LiteralValue::Unknown,
    ]);
    let optioned = run_with_literals(entry, options.clone(), literals.clone());
    assert!(
        optioned.diagnostics.is_empty(),
        "{:?}",
        optioned.diagnostics
    );

    let mut bad = options.clone();
    bad[5] = double(&[1, 1]);
    bad[7] = object(SIGNAL_INTERPOLATION);
    let bad = codes(&run_with_literals(entry, bad, literals.clone()));
    assert!(bad.contains(&(ARGUMENTS.into(), Some(5))) && bad.contains(&(TYPED.into(), Some(7))));

    let mut unknown_name = literals.clone();
    unknown_name[6] = LiteralValue::String("Tolerance".into());
    assert!(
        codes(&run_with_literals(entry, options.clone(), unknown_name))
            .contains(&(OPTION.into(), Some(6)))
    );

    let mut duplicate = literals;
    duplicate[6] = LiteralValue::String("ARGUMENTS".into());
    assert!(
        codes(&run_with_literals(entry, options, duplicate)).contains(&(OPTION.into(), Some(6)))
    );

    let mut not_callable = base.clone();
    not_callable[1] = text();
    not_callable[2] = object(SIGNAL_SAMPLING);
    let not_callable = codes(&run(entry, not_callable));
    assert!(
        not_callable.contains(&(CALLABLE.into(), Some(1)))
            && not_callable.contains(&(TYPED.into(), Some(2)))
    );

    let mut odd = base.clone();
    odd.push(text());
    assert!(codes(&run(entry, odd))
        .iter()
        .any(|(code, _)| code == ARITY));
    assert!(codes(&run(entry, base[..3].to_vec()))
        .iter()
        .any(|(code, _)| code == ARITY));
    let unknown = vec![ValueFact::unknown(DynamicReason::RuntimeValue); 6];
    assert!(run(entry, unknown).diagnostics.is_empty());
}

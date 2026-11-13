use super::*;
use runmat_types::*;
#[test]
fn observers_have_distinct_workspace_ui_contracts_and_logical_scalar_outputs() {
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                Default::default(),
                true,
            ))),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("init".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let save = inference::infer_save(&request, &SAVEMILPSOLUTIONS_CATALOG_ENTRY);
    assert!(save.diagnostics.is_empty());
    assert_eq!(save.outputs[0], ValueFact::scalar(ValueKindFact::Logical));
    assert!(save.effects.0.contains(&EffectKind::WorkspaceRead));
    assert!(save.effects.0.contains(&EffectKind::WorkspaceWrite));
    assert!(!save.effects.0.contains(&EffectKind::UserInterface));
    let plot = inference::infer_plot(&request, &OPTIMPLOTMILP_CATALOG_ENTRY);
    assert!(
        !plot.diagnostics.is_empty(),
        "complete metadata cannot omit required bounds/time"
    );
    assert!(plot.effects.0.contains(&EffectKind::UserInterface));
    assert!(!plot.effects.0.contains(&EffectKind::WorkspaceWrite));
    request.literals.literal_args[2] = LiteralValue::String("unsupported".into());
    assert!(
        !inference::infer_save(&request, &SAVEMILPSOLUTIONS_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    for entry in ENTRIES {
        assert_eq!(entry.contract.maturity, BuiltinContractMaturity::Complete);
        assert_eq!(entry.documentation.examples.len(), 2);
        assert!(entry.documentation.example_exemption.is_none());
    }
}
#[test]
fn milp_callback_names_do_not_broaden_other_solver_profiles_or_allow_nested_cells() {
    let scalar = ValueFact::scalar(ValueKindFact::String);
    assert!(callback_metadata(&scalar));
    let inner = ValueFact::scalar(ValueKindFact::Cell(CellFact {
        element: Box::new(scalar.clone()),
        elements: vec![scalar.clone()],
        elements_complete: true,
    }));
    let nested = ValueFact::scalar(ValueKindFact::Cell(CellFact {
        element: Box::new(inner.clone()),
        elements: vec![inner],
        elements_complete: true,
    }));
    assert!(!callback_metadata(&nested));
    let mut request = CallRequest {
        arguments: vec![scalar.clone(), scalar.clone(), scalar],
        literals: LiteralContext::new(vec![
            LiteralValue::String("intlinprog".into()),
            LiteralValue::String("OutputFcn".into()),
            LiteralValue::String("savemilpsolutions".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let entry = &crate::OPTIMOPTIONS_CATALOG_ENTRY;
    let BuiltinInferenceRule::Identity(rule) = entry.contract.inference_rule else {
        panic!("identity-owned options inference");
    };
    assert!(rule.infer(&request, entry).diagnostics.is_empty());
    request.literals.literal_args[2] = LiteralValue::String("optimplotmilp".into());
    assert!(!rule.infer(&request, entry).diagnostics.is_empty());
    request.literals.literal_args[0] = LiteralValue::String("fminunc".into());
    assert!(!rule.infer(&request, entry).diagnostics.is_empty());
}

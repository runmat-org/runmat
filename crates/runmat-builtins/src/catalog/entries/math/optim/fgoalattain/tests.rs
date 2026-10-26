use super::*;
use runmat_types::*;
#[test]
fn canonical_shapes_stopped_frontiers_and_selected_options_are_truthful() {
    let numeric = || super::super::program_facts::double(ShapeFact::Unknown, StorageFact::Dense);
    let x = super::super::program_facts::double(
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    let mut args = vec![ValueFact::unknown(DynamicReason::RuntimeValue), x.clone()];
    args.extend([numeric(), numeric()]);
    let mut options = std::collections::BTreeMap::new();
    options.insert(
        "EqualityGoalCount".into(),
        super::super::nonlinear_facts::scalar(),
    );
    for _ in 0..7 {
        args.push(ValueFact::unknown(DynamicReason::RuntimeValue));
    }
    args.push(ValueFact::scalar(ValueKindFact::Struct(
        StructFact::scalar(options, true),
    )));
    let request = CallRequest {
        arguments: args,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(6)),
    };
    let result = inference::infer(&request, &FGOALATTAIN_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, x.shape);
    assert_eq!(result.outputs[1].shape, ShapeFact::Unknown);
    assert_eq!(
        result.outputs[2].shape,
        ShapeFact::Unknown,
        "unavailable objective/factor outputs can be empty"
    );
    assert!(result.effects.0.contains(&EffectKind::Unknown));
    assert!(!result.effects.0.contains(&EffectKind::Clock));
    assert_eq!(
        crate::OptimSolver::parse("fgoalattain"),
        Some(crate::OptimSolver::Fgoalattain)
    );
    assert!(crate::OptimOptionField::parse_for_solver(
        "Algorithm",
        crate::OptimSolver::Fgoalattain
    )
    .is_none());
    assert_eq!(
        crate::OptimOptionField::parse_for_solver(
            "GoalsExactAchieve",
            crate::OptimSolver::Fgoalattain
        ),
        Some(crate::OptimOptionField::EqualityGoalCount)
    );
}

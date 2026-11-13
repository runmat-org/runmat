use super::*;
use runmat_types::*;
#[test]
fn forms_profiles_errors_and_observable_services_are_canonical() {
    assert_eq!(SURROGATEOPT_DESCRIPTOR.signatures.len(), 9);
    for signature in SURROGATEOPT_DESCRIPTOR.signatures {
        assert_eq!(signature.outputs.len(), 5);
    }
    assert_eq!(
        SURROGATEOPT_ERROR_CHECKPOINT.identifier,
        Some("RunMat:surrogateopt:Checkpoint")
    );
    for effect in [
        EffectKind::FilesystemRead,
        EffectKind::FilesystemWrite,
        EffectKind::UserInterface,
        EffectKind::Randomness,
        EffectKind::Clock,
        EffectKind::HostCallback,
    ] {
        assert!(SURROGATEOPT_CATALOG_ENTRY
            .contract
            .effects
            .contains(&effect));
    }
    for field in [
        crate::OptimOptionField::MaxFunEvals,
        crate::OptimOptionField::BatchUpdateInterval,
        crate::OptimOptionField::CheckpointFile,
        crate::OptimOptionField::MinSurrogatePoints,
        crate::OptimOptionField::MinSampleDistance,
    ] {
        assert_eq!(
            crate::OptimOptionField::parse_for_solver(
                field.name_for_solver(crate::OptimSolver::Surrogateopt),
                crate::OptimSolver::Surrogateopt
            ),
            Some(field)
        );
    }
    assert!(!crate::OptimSolver::Surrogateopt
        .accepts(crate::OptimOptionField::SpecifyObjectiveGradient));
    assert!(documentation::SOLVER
        .examples
        .iter()
        .all(|example| matches!(
            example.verification,
            crate::BuiltinExampleVerification::Assertions { .. }
        )));
}
#[test]
fn inference_keeps_empty_or_absent_results_dynamic_and_checks_double_data() {
    let double = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            double.clone(),
            double.clone(),
        ],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 3]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
    };
    let result = inference::solver(&request, &SURROGATEOPT_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 5);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(result.outputs[1].shape, ShapeFact::Unknown);
    let ValueKindFact::Struct(trials) = &result.outputs[4].kind else {
        panic!("trials");
    };
    assert!(
        trials.fields_complete
            && ["X", "Fval", "Ineq"]
                .iter()
                .all(|name| trials.fields.contains_key(*name))
    );
    let mut invalid = request;
    invalid.arguments[1].kind = ValueKindFact::Logical;
    assert!(!inference::solver(&invalid, &SURROGATEOPT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let checkpoint = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::new(vec![LiteralValue::Unknown]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
    };
    assert!(inference::solver(&checkpoint, &SURROGATEOPT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

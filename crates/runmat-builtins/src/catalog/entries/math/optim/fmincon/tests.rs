use super::*;
use runmat_types::*;
#[test]
fn canonical_sqp_outputs_keep_empty_frontiers_and_callback_clock_authority() {
    assert_eq!(
        crate::OptimAlgorithm::parse("active-set"),
        Some(crate::OptimAlgorithm::ActiveSet)
    );
    assert!(crate::OptimSolver::Fmincon.accepts_algorithm(crate::OptimAlgorithm::ActiveSet));
    assert!(crate::OptimSolver::Quadprog.accepts_algorithm(crate::OptimAlgorithm::ActiveSet));
    assert_eq!(crate::OptimAlgorithm::ActiveSet.name(), "active-set");
    let empty = || {
        super::super::program_facts::double(
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        )
    };
    let x = super::super::program_facts::double(
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    let mut options = std::collections::BTreeMap::new();
    options.insert("Algorithm".into(), ValueFact::scalar(ValueKindFact::String));
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            x.clone(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(options, true))),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(7)),
    };
    let result = inference::infer(&request, &FMINCON_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, x.shape);
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::Unknown,
        "fval may be [] when no evaluation is permitted"
    );
    assert_eq!(
        result.outputs[5].shape,
        ShapeFact::Unknown,
        "a stopped derivative sweep cannot fabricate a gradient"
    );
    assert!(result.effects.0.contains(&EffectKind::Unknown));
    assert!(result.effects.0.contains(&EffectKind::Clock));
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(3));
    assert!(!inference::infer(&request, &FMINCON_CATALOG_ENTRY)
        .effects
        .0
        .contains(&EffectKind::Clock));
    assert_eq!(
        crate::OptimSolver::parse("fmincon"),
        Some(crate::OptimSolver::Fmincon)
    );
    assert_eq!(
        crate::OptimOptionField::parse_for_solver("GradConstr", crate::OptimSolver::Fmincon),
        Some(crate::OptimOptionField::SpecifyConstraintGradient)
    );
    assert_eq!(
        crate::OptimOptionField::parse_for_solver("HessianFcn", crate::OptimSolver::Fmincon),
        Some(crate::OptimOptionField::HessianFcn)
    );
    assert!(
        crate::OptimSolver::Fmincon.accepts_algorithm(crate::OptimAlgorithm::TrustRegionReflective)
    );
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(7));
    let reflected = inference::infer(&request, &FMINCON_CATALOG_ENTRY);
    assert_eq!(reflected.outputs[6].storage, StorageFact::Unknown);
    let ValueKindFact::Struct(info) = &reflected.outputs[3].kind else {
        panic!("information struct");
    };
    assert!(
        !info.fields_complete,
        "algorithm-specific fields remain possible"
    );
    let ValueKindFact::Struct(options) = &mut request.arguments[9].kind else {
        panic!("options struct");
    };
    options.fields.insert(
        "HessianMultiplyFcn".into(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    assert!(matches!(
        inference::infer(&request, &FMINCON_CATALOG_ENTRY).outputs[6].kind,
        ValueKindFact::Unknown
    ));
}

use super::*;
use crate::*;
use runmat_types::*;
#[test]
fn typed_milp_profile_ranges_are_distinct_from_lp_iteration_options() {
    use OptimOptionField as F;
    for field in [
        F::AbsoluteGapTolerance,
        F::RelativeGapTolerance,
        F::MaxNodes,
        F::MaxFeasiblePoints,
        F::MaxTime,
        F::ObjectiveCutOff,
        F::Display,
        F::TolCon,
        F::OutputFcn,
        F::PlotFcn,
    ] {
        assert!(OptimSolver::Intlinprog.accepts(field));
        assert!(OptimOptionField::ALL.contains(&field));
    }
    assert!(!OptimSolver::Intlinprog.accepts(F::Algorithm));
    assert!(!OptimSolver::Intlinprog.accepts(F::MaxIter));
    assert!(!OptimSolver::Linprog.accepts(F::MaxNodes));
    assert!(F::RelativeGapTolerance.intlinprog_scalar_range(0.0));
    assert!(!F::RelativeGapTolerance.intlinprog_scalar_range(1.1));
    assert!(!F::AbsoluteGapTolerance.intlinprog_scalar_range(f64::INFINITY));
    assert!(F::MaxTime.intlinprog_scalar_range(0.0));
    assert!(F::MaxTime.intlinprog_scalar_range(f64::INFINITY));
    assert!(F::MaxFeasiblePoints.intlinprog_scalar_range(f64::INFINITY));
    assert!(!F::MaxNodes.intlinprog_scalar_range(f64::INFINITY));
    assert!(!F::MaxNodes.intlinprog_scalar_range(1.5));
    assert!(!F::ObjectiveCutOff.intlinprog_scalar_range(f64::NEG_INFINITY));
    assert!(!F::TolCon.intlinprog_scalar_range(0.0));
    assert!(INTLINPROG_INTEGER_CAPABILITIES
        .iter()
        .flat_map(|c| c.inputs)
        .all(|input| input.availability == BuiltinIntegerInputAvailability::RunMatOnly));
}
#[test]
fn outputs_and_clock_are_truthful_for_early_failure_and_actual_deadlines() {
    let numeric = |shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    };
    let mut request = CallRequest {
        arguments: vec![
            numeric(vec![Some(2), Some(1)]),
            numeric(vec![Some(1), Some(2)]),
            numeric(vec![Some(1), Some(2)]),
            numeric(vec![Some(1), Some(1)]),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(4)),
    };
    let result = inference::infer(&request, &INTLINPROG_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 4);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::Unknown,
        "terminal failures can return []"
    );
    let ValueKindFact::Struct(info) = &result.outputs[3].kind else {
        panic!("closed information schema")
    };
    assert!(info.fields_complete);
    assert_eq!(info.fields.len(), 8);
    for key in [
        "relativegap",
        "absolutegap",
        "numfeaspoints",
        "numnodes",
        "constrviolation",
        "message",
        "lowerbound",
        "certifiedobjective",
    ] {
        assert!(info.fields.contains_key(key));
    }
    assert!(result.effects.0.contains(&EffectKind::Clock));
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(1));
    assert!(
        inference::infer(&request, &INTLINPROG_CATALOG_ENTRY)
            .effects
            .0
            .contains(&EffectKind::Clock),
        "MaxTime is observable even without a diagnostics output"
    );
    request.arguments[3] = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    assert!(!inference::infer(&request, &INTLINPROG_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

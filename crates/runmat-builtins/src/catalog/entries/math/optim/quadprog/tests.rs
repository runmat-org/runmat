use super::*;
use crate::*;
use runmat_types::*;
#[test]
fn qp_facts_preserve_conditional_outputs_and_distinct_controls() {
    let real = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    let numeric =
        |shape| ValueFact::proven(real.clone(), ShapeFact::from(shape), StorageFact::Dense);
    let mut request = CallRequest {
        arguments: vec![
            numeric(vec![Some(2), Some(2)]),
            numeric(vec![Some(2), Some(1)]),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
    };
    let inference = inference::infer(&request, &QUADPROG_CATALOG_ENTRY);
    assert!(inference.diagnostics.is_empty());
    assert!(inference.effects.0.contains(&EffectKind::Clock));
    let mut unobserved = request.clone();
    unobserved.outputs = OutputSelection::new(RequestedOutputCount::Exactly(3));
    assert!(!inference::infer(&unobserved, &QUADPROG_CATALOG_ENTRY)
        .effects
        .0
        .contains(&EffectKind::Clock));
    unobserved.outputs = OutputSelection::new(RequestedOutputCount::CurrentFunctionNargout);
    assert!(inference::infer(&unobserved, &QUADPROG_CATALOG_ENTRY)
        .effects
        .0
        .contains(&EffectKind::Clock));
    assert_eq!(inference.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(inference.outputs[4].kind, ValueKindFact::Unknown);
    let mut warm = request.clone();
    warm.arguments
        .extend(std::iter::repeat_n(numeric(vec![Some(0), Some(0)]), 6));
    warm.arguments.push(ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(QUADPROG_WARM_START_CLASS.owned()),
            properties: std::collections::BTreeMap::from([
                ("X".into(), numeric(vec![Some(2), Some(1)])),
                (
                    "Options".into(),
                    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                        std::collections::BTreeMap::from([(
                            "MaxTime".into(),
                            numeric(vec![Some(1), Some(1)]),
                        )]),
                        true,
                    ))),
                ),
            ]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    ));
    warm.outputs = OutputSelection::new(RequestedOutputCount::Exactly(3));
    let selected = inference::infer(&warm, &QUADPROG_CATALOG_ENTRY);
    assert!(selected.diagnostics.is_empty());
    assert!(
        selected.effects.0.contains(&EffectKind::Clock),
        "finite-capable MaxTime is not erased by an unobserved output"
    );
    assert!(
        matches!(&selected.outputs[0].kind,ValueKindFact::Object(object)
        if object.runtime_class.as_ref().is_some_and(|class|class.is(QUADPROG_WARM_START_CLASS)))
    );
    if let ValueKindFact::Object(object) = &mut warm.arguments[8].kind {
        object.runtime_class = None;
    }
    let dynamic = inference::infer(&warm, &QUADPROG_CATALOG_ENTRY);
    assert!(dynamic.diagnostics.is_empty());
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Unknown);
    request.arguments[0] = numeric(vec![Some(2), Some(3)]);
    assert!(!inference::infer(&request, &QUADPROG_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    assert!(OptimSolver::Quadprog.accepts(OptimOptionField::TolCon));
    assert!(OptimSolver::Quadprog.accepts_algorithm(OptimAlgorithm::ActiveSet));
    assert!(OptimSolver::Quadprog.accepts(OptimOptionField::MaxTime));
    assert!(!OptimSolver::Coneprog.accepts(OptimOptionField::TolCon));
    assert_eq!(
        OptimOptionField::parse_for_solver("ConstraintTolerance", OptimSolver::Quadprog),
        Some(OptimOptionField::TolCon)
    );
    assert_eq!(
        OptimOptionField::parse_for_solver("FunctionTolerance", OptimSolver::Quadprog),
        None
    );
    assert!(!OptimSolver::Quadprog.accepts_algorithm(OptimAlgorithm::TrustRegionReflective));
}

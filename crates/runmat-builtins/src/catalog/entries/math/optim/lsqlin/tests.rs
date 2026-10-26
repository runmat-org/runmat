use super::*;
use crate::*;
use runmat_types::*;
#[test]
fn rectangular_original_geometry_and_six_output_contract() {
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
            numeric(vec![Some(3), Some(2)]),
            numeric(vec![Some(3), Some(1)]),
            numeric(vec![Some(0), Some(0)]),
            numeric(vec![Some(0), Some(0)]),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(6)),
    };
    let result = inference::infer(&request, &LSQLIN_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert!(result.effects.0.contains(&EffectKind::Clock));
    let mut unobserved = request.clone();
    unobserved.outputs = OutputSelection::new(RequestedOutputCount::Exactly(4));
    assert!(!inference::infer(&unobserved, &LSQLIN_CATALOG_ENTRY)
        .effects
        .0
        .contains(&EffectKind::Clock));
    unobserved.outputs = OutputSelection::new(RequestedOutputCount::CurrentFunctionNargout);
    assert!(inference::infer(&unobserved, &LSQLIN_CATALOG_ENTRY)
        .effects
        .0
        .contains(&EffectKind::Clock));
    assert_eq!(result.outputs.len(), 6);
    assert_eq!(result.outputs[2].shape, ShapeFact::Unknown);
    let mut array_start = request.clone();
    array_start
        .arguments
        .extend(std::iter::repeat_n(numeric(vec![Some(0), Some(0)]), 4));
    array_start
        .arguments
        .push(numeric(vec![Some(1), Some(1), Some(1), Some(2)]));
    assert!(inference::infer(&array_start, &LSQLIN_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let mut warm = request.clone();
    warm.arguments
        .extend(std::iter::repeat_n(numeric(vec![Some(0), Some(0)]), 4));
    warm.arguments.push(ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(LSQLIN_WARM_START_CLASS.owned()),
            properties: std::collections::BTreeMap::from([
                ("X".into(), numeric(vec![Some(2), Some(1)])),
                (
                    "Options".into(),
                    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                        std::collections::BTreeMap::new(),
                        false,
                    ))),
                ),
            ]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    ));
    let inferred = inference::infer(&warm, &LSQLIN_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert!(
        matches!(&inferred.outputs[0].kind, ValueKindFact::Object(object)
        if object.runtime_class.as_ref().is_some_and(|class|class.is(LSQLIN_WARM_START_CLASS)))
    );
    assert!(inferred.effects.0.contains(&EffectKind::Clock));
    request.arguments[1] = numeric(vec![Some(2), Some(1)]);
    assert!(!inference::infer(&request, &LSQLIN_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
#[test]
fn documented_controls_and_compatibility_are_explicit() {
    assert!(OptimSolver::Lsqlin.accepts(OptimOptionField::TolCon));
    assert!(OptimSolver::Lsqlin.accepts(OptimOptionField::TolX));
    assert!(OptimSolver::Lsqlin.accepts(OptimOptionField::ObjectiveLimit));
    assert_eq!(
        OptimOptionField::parse_for_solver("FunctionTolerance", OptimSolver::Lsqlin),
        Some(OptimOptionField::FunctionTolerance)
    );
    assert!(OptimSolver::Lsqlin.accepts_algorithm(OptimAlgorithm::TrustRegionReflective));
    assert!(OptimSolver::Lsqlin.accepts_algorithm(OptimAlgorithm::ActiveSet));
    assert!(OptimSolver::Lsqlin.accepts(OptimOptionField::TypicalX));
    assert!(OptimSolver::Lsqlin.accepts(OptimOptionField::MaxPCGIter));
    assert!(OptimSolver::Lsqlin.accepts_algorithm(OptimAlgorithm::InteriorPoint));
    assert!(!OptimSolver::Lsqlin.accepts_algorithm(OptimAlgorithm::InteriorPointConvex));
    assert_eq!(LSQLIN_CATALOG_ENTRY.extensions.len(), 11);
    assert!(LSQLIN_INTEGER_CAPABILITIES
        .iter()
        .flat_map(|c| c.inputs)
        .all(|c| c.availability == BuiltinIntegerInputAvailability::RunMatOnly));
}

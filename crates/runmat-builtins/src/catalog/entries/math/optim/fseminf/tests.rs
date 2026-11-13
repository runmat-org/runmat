use super::*;
use runmat_types::*;
#[test]
fn complete_public_forms_keep_shape_empty_frontiers_and_clock_authority() {
    assert_eq!(
        crate::OptimSolver::parse("fseminf"),
        Some(crate::OptimSolver::Fseminf)
    );
    let double = |shape| super::super::program_facts::double(shape, StorageFact::Dense);
    let empty = || double(ShapeFact::from(vec![Some(0), Some(0)]));
    let x = double(ShapeFact::from(vec![Some(2), Some(2)]));
    let mut args = vec![
        ValueFact::unknown(DynamicReason::RuntimeValue),
        x.clone(),
        double(ShapeFact::Scalar),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ];
    args.resize_with(10, empty);
    args.push(ValueFact::scalar(ValueKindFact::Struct(
        StructFact::scalar(Default::default(), true),
    )));
    for n in [4, 6, 8, 10, 11] {
        let request = CallRequest {
            arguments: args[..n].to_vec(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
        };
        let result = inference::infer(&request, &FSEMINF_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 5);
        assert_eq!(result.outputs[0].shape, x.shape);
        assert_eq!(result.outputs[1].shape, ShapeFact::Unknown);
        assert!(result.effects.0.contains(&EffectKind::Clock));
        let request = CallRequest {
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
            ..request
        };
        assert!(!inference::infer(&request, &FSEMINF_CATALOG_ENTRY)
            .effects
            .0
            .contains(&EffectKind::Clock));
    }
    assert_eq!(FSEMINF_DESCRIPTOR.signatures.len(), 6);
    assert!(FSEMINF_DESCRIPTOR
        .signatures
        .iter()
        .all(|s| s.outputs.len() == 5));
}
#[test]
fn semantic_option_profile_has_no_algorithm_or_derivative_placeholders() {
    use crate::OptimOptionField as F;
    use crate::OptimSolver as S;
    for name in [
        "DiffMinChange",
        "DiffMaxChange",
        "Diagnostics",
        "FunValCheck",
        "MaxSQPIter",
        "RelLineSrchBnd",
        "RelLineSrchBndDuration",
        "TolConSQP",
        "PlotFcns",
        "FinDiffRelStep",
        "TypicalX",
        "FunctionTolerance",
        "OptimalityTolerance",
    ] {
        assert!(F::parse_for_solver(name, S::Fseminf).is_some(), "{name}");
    }
    for field in [
        F::Algorithm,
        F::UseParallel,
        F::SpecifyConstraintGradient,
        F::MeritFunction,
    ] {
        assert!(!S::Fseminf.accepts(field));
    }
    for field in [F::DiffMaxChange, F::MaxSQPIter, F::TolConSQP] {
        assert!(!S::Fmincon.accepts(field));
    }
    assert_eq!(
        F::parse_for_solver("PlotFcns", S::Fseminf),
        Some(F::PlotFcn)
    );
    assert!(S::Fseminf.accepts_zero_iterations());
}

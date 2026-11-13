use super::*;
use runmat_types::*;
#[test]
fn canonical_forms_controls_and_helper_grammars_are_distinct() {
    assert_eq!(PARTICLESWARM_DESCRIPTOR.signatures.len(), 4);
    assert_eq!(PARTICLESWARM_DESCRIPTOR.signatures[0].inputs.len(), 2);
    assert_eq!(PARTICLESWARM_DESCRIPTOR.signatures[1].inputs.len(), 4);
    assert_eq!(PARTICLESWARM_DESCRIPTOR.signatures[2].inputs.len(), 5);
    assert_eq!(PARTICLESWARM_DESCRIPTOR.signatures[3].inputs.len(), 1);
    assert_eq!(PARTICLESWARM_DESCRIPTOR.signatures[0].outputs.len(), 5);
    assert_eq!(PSWCREATIONUNIFORM_DESCRIPTOR.signatures[0].inputs.len(), 1);
    assert_eq!(PSWPLOTBESTF_DESCRIPTOR.signatures[0].inputs.len(), 2);
    use crate::{OptimOptionField as F, OptimSolver as S};
    for (name, field) in [
        ("SwarmSize", F::SwarmSize),
        ("MaxIterations", F::MaxIter),
        ("InitialSwarmMatrix", F::InitialPoints),
        ("InitialSwarm", F::InitialPoints),
        ("UseVectorized", F::UseVectorized),
        ("HybridFcn", F::HybridFcn),
    ] {
        assert_eq!(F::parse_for_solver(name, S::Particleswarm), Some(field));
    }
    for name in [
        "Algorithm",
        "MaxFunctionEvaluations",
        "SpecifyObjectiveGradient",
        "TypicalX",
    ] {
        assert_eq!(F::parse_for_solver(name, S::Particleswarm), None);
    }
    assert_eq!(
        F::MaxIter.name_for_solver(S::Particleswarm),
        "MaxIterations"
    );
    assert_eq!(F::MaxIter.name_for_solver(S::Fmincon), "MaxIter");
    for effect in [
        EffectKind::Randomness,
        EffectKind::Clock,
        EffectKind::HostCallback,
    ] {
        assert!(PARTICLESWARM_CATALOG_ENTRY
            .contract
            .effects
            .contains(&effect));
    }
    assert_eq!(
        PARTICLESWARM_ERROR_INVALID_ARGUMENT.identifier,
        Some("RunMat:particleswarm:InvalidArgument")
    );
    assert_eq!(
        PSWPLOTBESTF_ERROR_INVALID_ARGUMENT.identifier,
        Some("RunMat:pswplotbestf:InvalidArgument")
    );
}
#[test]
fn inference_counts_host_rows_and_does_not_claim_bounded_inconsistent_results_have_a_point() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let double = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![unknown.clone(), double.clone()],
        literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
    };
    let result = inference::solver(&request, &PARTICLESWARM_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 5);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Struct(output) = &result.outputs[3].kind else {
        panic!("output")
    };
    for name in [
        "iterations",
        "funccount",
        "hybridflag",
        "message",
        "rngstate",
    ] {
        assert!(output.fields.contains_key(name));
    }
    let mut bounded = request.clone();
    bounded.arguments.extend([double.clone(), double]);
    let result = inference::solver(&bounded, &PARTICLESWARM_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    let mut invalid = request;
    invalid.arguments[1].kind = ValueKindFact::Logical;
    assert!(!inference::solver(&invalid, &PARTICLESWARM_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

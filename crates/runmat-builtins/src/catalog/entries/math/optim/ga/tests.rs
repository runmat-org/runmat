use super::*;
use runmat_types::*;
#[test]
fn every_registered_genetic_family_leaf_has_an_authored_contract_and_example() {
    assert_eq!(ENTRIES.len(), 39);
    let mut names = std::collections::BTreeSet::new();
    for entry in ENTRIES {
        assert!(names.insert(entry.identity.name));
        assert!(
            !entry.documentation.examples.is_empty(),
            "{}",
            entry.identity.name
        );
        assert!(entry.documentation.example_exemption.is_none());
        assert!(!entry.descriptor.signatures.is_empty());
        assert_ne!(
            entry.contract.maturity,
            crate::BuiltinContractMaturity::Incomplete
        );
        assert!(entry
            .descriptor
            .errors
            .iter()
            .any(|error| error.identifier == GA_ERROR_INVALID_ARGUMENT.identifier));
    }
    let state = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        std::collections::BTreeMap::new(),
        true,
    )));
    let request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            state.clone(),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("iter".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = plot_inference::infer(&request, &GAPLOTBESTF_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs, [state]);
}
#[test]
fn public_forms_and_typed_genetic_profile_keep_their_distinct_meanings() {
    assert_eq!(
        GA_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.inputs.len())
            .collect::<Vec<_>>(),
        [2, 4, 6, 8, 9, 10, 10, 11, 1]
    );
    assert!(GA_DESCRIPTOR
        .signatures
        .iter()
        .all(|signature| signature.outputs.len() == 6));
    use crate::{OptimOptionField as F, OptimSolver as S};
    for (name, field) in [
        ("MaxGenerations", F::MaxIter),
        ("MaxStallGenerations", F::MaxStallIterations),
        ("FitnessLimit", F::ObjectiveLimit),
        ("InitialPopulationMatrix", F::InitialPopulationMatrix),
        ("InitialScores", F::InitialScoresMatrix),
        ("PopInitRange", F::InitialPopulationRange),
        ("ConstraintTolerance", F::TolCon),
        ("UseVectorized", F::UseVectorized),
    ] {
        assert_eq!(F::parse_for_solver(name, S::Ga), Some(field));
        assert!(S::Ga.accepts(field));
    }
    for field in [
        F::Algorithm,
        F::MaxFunEvals,
        F::SpecifyObjectiveGradient,
        F::HessianMultiplyFcn,
    ] {
        assert!(!S::Ga.accepts(field));
    }
    assert_eq!(F::MaxIter.name_for_solver(S::Ga), "MaxGenerations");
    assert_eq!(F::ObjectiveLimit.name_for_solver(S::Ga), "FitnessLimit");
    assert_eq!(
        GA_ERROR_INVALID_ARGUMENT.identifier,
        Some("RunMat:ga:InvalidArgument")
    );
    for effect in [
        EffectKind::Randomness,
        EffectKind::Clock,
        EffectKind::HostCallback,
        EffectKind::Unknown,
    ] {
        assert!(GA_CATALOG_ENTRY.contract.effects.contains(&effect));
    }
    assert_eq!(GA_CATALOG_ENTRY.extensions.len(), 3);
    assert!(!GA_CATALOG_ENTRY.contract.effect_set().is_side_effect_free());
}
#[test]
fn inference_never_manufactures_a_nonempty_point_or_custom_genome_numeric_class() {
    let double = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue), double],
        literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.)]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(6)),
    };
    let result = inference::solver(&request, &GA_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 6);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Struct(output) = &result.outputs[3].kind else {
        panic!("fixed output schema");
    };
    assert!(output.fields_complete);
    for field in [
        "generations",
        "objectiveCallCount",
        "nonlinearCallCount",
        "innergenerations",
        "linearnodes",
        "fitnessprovenance",
        "hybridaccepted",
        "totaltime",
    ] {
        assert!(output.fields.contains_key(field));
    }
    let mut invalid = request.clone();
    invalid.arguments[1].kind = ValueKindFact::Logical;
    assert!(!inference::solver(&invalid, &GA_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    invalid = request.clone();
    invalid.literals = LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(0.)]);
    assert!(!inference::solver(&invalid, &GA_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let mut excessive = request;
    excessive.outputs = OutputSelection::new(RequestedOutputCount::Exactly(7));
    assert!(!inference::solver(&excessive, &GA_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

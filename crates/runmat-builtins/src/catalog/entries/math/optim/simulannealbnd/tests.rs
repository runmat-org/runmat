use super::*;
use runmat_types::*;
#[test]
fn canonical_forms_options_effects_and_actual_helpers_remain_distinct() {
    let signatures = SIMULANNEALBND_DESCRIPTOR.signatures;
    assert_eq!(signatures.len(), 4);
    assert_eq!(
        signatures
            .iter()
            .map(|s| s.inputs.len())
            .collect::<Vec<_>>(),
        vec![2, 4, 5, 1]
    );
    assert!(signatures.iter().all(|s| s.outputs.len() == 4));
    use crate::{OptimOptionField as F, OptimSolver as S};
    for (name, field) in [
        ("AcceptanceFcn", F::AcceptanceFcn),
        ("AnnealingFcn", F::AnnealingFcn),
        ("DataType", F::DataType),
        ("InitialTemperature", F::InitialTemperature),
        ("TemperatureFcn", F::TemperatureFcn),
        ("ReannealInterval", F::ReannealInterval),
        ("HybridInterval", F::HybridInterval),
        ("PlotInterval", F::PlotInterval),
        ("MaxIterations", F::MaxIter),
        ("MaxFunctionEvaluations", F::MaxFunEvals),
        ("TolFun", F::FunctionTolerance),
        ("OutputFcns", F::OutputFcn),
        ("StallIterLimit", F::MaxStallIterations),
        ("TimeLimit", F::MaxTime),
    ] {
        assert_eq!(F::parse_for_solver(name, S::Simulannealbnd), Some(field));
    }
    for name in ["UseParallel", "UseVectorized", "TypicalX", "Algorithm"] {
        assert_eq!(F::parse_for_solver(name, S::Simulannealbnd), None);
    }
    for effect in [
        EffectKind::Randomness,
        EffectKind::Clock,
        EffectKind::HostCallback,
    ] {
        assert!(SIMULANNEALBND_CATALOG_ENTRY
            .contract
            .effects
            .contains(&effect));
    }
    assert!(S::Simulannealbnd.accepts_display(crate::OptimDisplay::Diagnose));
    assert!(!S::Simulannealbnd.accepts_display(crate::OptimDisplay::None));
    assert!(!S::Simulannealbnd.accepts_display(crate::OptimDisplay::Notify));
    assert_eq!(ACCEPTANCESA_DESCRIPTOR.signatures[0].inputs.len(), 3);
    assert_eq!(ANNEALINGFAST_DESCRIPTOR.signatures[0].inputs.len(), 2);
    assert_eq!(TEMPERATUREEXP_DESCRIPTOR.signatures[0].inputs.len(), 2);
    assert_eq!(SAPLOTBESTF_DESCRIPTOR.signatures[0].inputs.len(), 3);
}
#[test]
fn result_inference_preserves_double_shape_but_does_not_forge_custom_payload_shape() {
    let x = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let mut request = CallRequest {
        arguments: vec![unknown.clone(), x.clone()],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(4)),
    };
    let result = inference::solver(&request, &SIMULANNEALBND_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 4);
    assert_eq!(result.outputs[0].shape, x.shape);
    let ValueKindFact::Struct(output) = &result.outputs[3].kind else {
        panic!("output")
    };
    for name in [
        "iterations",
        "funccount",
        "message",
        "problemtype",
        "temperature",
        "totaltime",
        "rngstate",
    ] {
        assert!(output.fields.contains_key(name));
    }
    request
        .arguments
        .extend([unknown.clone(), unknown.clone(), unknown]);
    request
        .literals
        .literal_args
        .resize(5, LiteralValue::Unknown);
    let result = inference::solver(&request, &SIMULANNEALBND_CATALOG_ENTRY);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("datatype".into(), ValueFact::scalar(ValueKindFact::String));
    request.arguments[4] =
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let opaque = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        std::collections::BTreeMap::new(),
        true,
    )));
    request.arguments[1] = opaque;
    let result = inference::solver(&request, &SIMULANNEALBND_CATALOG_ENTRY);
    assert!(
        result.diagnostics.is_empty(),
        "lowercase custom selector must not force double x0"
    );
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}

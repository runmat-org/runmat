use super::*;
use runmat_types::*;

#[test]
fn conic_outputs_and_problem_fields_remain_conditional() {
    let real = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let empty = ValueFact::proven(
        real.kind.clone(),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    let mut request = CallRequest {
        arguments: vec![real, empty],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
    };
    let result = inference::infer(&request, &CONEPROG_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 5);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    // Lambda is a numeric-column struct only at verified optimum, but [] for
    // terminal failure. An unconditional structure/cell proof is false.
    assert_eq!(result.outputs[4].kind, ValueKindFact::Unknown);
    // The second argument is the cone descriptor, not a trailing options struct.
    request.arguments[1] = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        Default::default(),
        false,
    )));
    assert!(inference::infer(&request, &CONEPROG_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let numeric = |shape| {
        ValueFact::proven(
            request.arguments[0].kind.clone(),
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    };
    let mut descriptor = std::collections::BTreeMap::new();
    descriptor.insert("A".into(), numeric(vec![Some(2), Some(1)]));
    descriptor.insert("b".into(), numeric(vec![Some(2), Some(1)]));
    descriptor.insert("d".into(), numeric(vec![Some(1), Some(1)]));
    descriptor.insert("gamma".into(), request.arguments[0].clone());
    request.arguments[1] = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        descriptor.clone(),
        true,
    )));
    assert!(inference::infer(&request, &CONEPROG_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    for (field, shape) in [
        ("b", vec![Some(1), Some(1)]),
        ("d", vec![Some(2), Some(1)]),
        ("A", vec![Some(2), Some(2)]),
    ] {
        let mut invalid = descriptor.clone();
        invalid.insert(
            field.into(),
            ValueFact::proven(
                request.arguments[0].kind.clone(),
                ShapeFact::from(shape),
                StorageFact::Dense,
            ),
        );
        if field == "A" {
            invalid.get_mut("d").unwrap().shape = ShapeFact::from(vec![Some(2), Some(1)]);
        }
        request.arguments[1] =
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(invalid, true)));
        assert!(
            !inference::infer(&request, &CONEPROG_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            "nested {field} geometry"
        );
    }
    for fact in descriptor.values_mut() {
        fact.shape = ShapeFact::Unknown;
    }
    request.arguments[1] =
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(descriptor, true)));
    assert!(
        inference::infer(&request, &CONEPROG_CATALOG_ENTRY)
            .diagnostics
            .is_empty(),
        "unknown geometry remains dynamic"
    );
    request.arguments = vec![ValueFact::scalar(ValueKindFact::Struct(
        StructFact::scalar(Default::default(), true),
    ))];
    assert!(!inference::infer(&request, &CONEPROG_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn conic_control_profile_and_effects_describe_real_consumers() {
    use crate::{OptimOptionField, OptimSolver};
    assert!(OptimSolver::Coneprog.accepts_zero_iterations());
    for field in [
        OptimOptionField::TolX,
        OptimOptionField::TolFun,
        OptimOptionField::MaxIter,
        OptimOptionField::Display,
        OptimOptionField::Algorithm,
    ] {
        assert!(OptimSolver::Coneprog.accepts(field));
    }
    assert!(!OptimSolver::Coneprog.accepts(OptimOptionField::MaxFunEvals));
    assert!(!OptimSolver::Coneprog.accepts(OptimOptionField::OutputFcn));
    let real = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let empty = ValueFact::proven(
        real.kind.clone(),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    let mut request = CallRequest {
        arguments: vec![real, empty.clone()],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(4)),
    };
    let quiet = inference::infer(&request, &CONEPROG_CATALOG_ENTRY);
    assert!(quiet.diagnostics.is_empty());
    assert!(!quiet.effects.0.contains(&EffectKind::HostCallback));
    request.arguments.push(empty);
    assert!(inference::infer(&request, &CONEPROG_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("Display".into(), ValueFact::scalar(ValueKindFact::String));
    request.arguments[2] =
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let dynamic = inference::infer(&request, &CONEPROG_CATALOG_ENTRY);
    assert!(dynamic.effects.0.contains(&EffectKind::HostCallback));
    let ValueKindFact::Struct(output) = &dynamic.outputs[3].kind else {
        panic!("output")
    };
    assert!(output.fields.contains_key("linearsolver") && output.fields.contains_key("dualitygap"));
}

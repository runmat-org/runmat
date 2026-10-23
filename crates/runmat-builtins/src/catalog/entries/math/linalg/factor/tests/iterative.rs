use super::*;

#[test]
fn gmres_facts_effects_optional_controls_and_default_binding() {
    use runmat_types::EffectKind;
    for class in [NumericClass::Single, NumericClass::Double] {
        for domain in [NumericDomain::Real, NumericDomain::Complex] {
            let kind = ValueKindFact::Numeric(NumericFact { class, domain });
            let mut a = matrix(3, 3);
            a.kind = kind.clone();
            let mut b = matrix(3, 1);
            b.kind = kind.clone();
            let request = CallRequest {
                arguments: vec![a, b],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
            };
            let inferred = infer_catalog_call(&GMRES_CATALOG_ENTRY, &request);
            assert!(
                inferred.diagnostics.is_empty(),
                "{:?}",
                inferred.diagnostics
            );
            assert_eq!(inferred.outputs[0].kind, kind);
            assert_eq!(inferred.outputs[0].shape, request.arguments[1].shape);
            assert_eq!(
                inferred.outputs[3].shape,
                ShapeFact::from(vec![Some(1), Some(2)])
            );
            assert!(!inferred.effects.0.contains(&EffectKind::Unknown));
            assert!(!inferred.effects.0.contains(&EffectKind::HostCallback));
        }
    }
    let mut request = CallRequest {
        arguments: vec![matrix(3, 3), matrix(3, 1), matrix(1, 1)],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(0.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
    };
    assert!(infer_catalog_call(&GMRES_CATALOG_ENTRY, &request)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GMRES.INVALID_INPUT"));
    request.literals.literal_args[2] = LiteralValue::Number(1.0);
    request.arguments[0] = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    assert!(infer_catalog_call(&GMRES_CATALOG_ENTRY, &request)
        .effects
        .0
        .contains(&EffectKind::Unknown));
    request.arguments.push(matrix(1, 1));
    request.arguments[3].residency = ResidencyFact::Device { provider: None };
    assert!(infer_catalog_call(&GMRES_CATALOG_ENTRY, &request)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.GMRES.UNSUPPORTED_INPUT"));
    assert_eq!(
        GMRES_CATALOG_ENTRY.bindings,
        &crate::REQUIRED_DEFAULT_BINDING
    );
    assert_eq!(
        GMRES_CATALOG_ENTRY.placement.accelerator,
        crate::BuiltinAcceleratorPolicy::Forbidden
    );
}

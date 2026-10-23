use super::*;

#[test]
fn identity_controls_output_shape_arity_and_effects_without_name_dispatch() {
    for (entry, minimum, handles) in [
        (&crate::LINE_CATALOG_ENTRY, 0, Handles::Series),
        (&crate::PLOT3_CATALOG_ENTRY, 3, Handles::Scalar),
        (&crate::SEMILOGX_CATALOG_ENTRY, 1, Handles::Scalar),
        (&crate::SEMILOGY_CATALOG_ENTRY, 1, Handles::Scalar),
        (&crate::LOGLOG_CATALOG_ENTRY, 1, Handles::Scalar),
        (&crate::SCATTER_CATALOG_ENTRY, 2, Handles::Scalar),
        (&crate::SCATTER3_CATALOG_ENTRY, 3, Handles::Scalar),
        (&crate::STAIRS_CATALOG_ENTRY, 1, Handles::Scalar),
        (&crate::STEM_CATALOG_ENTRY, 1, Handles::Scalar),
        (&crate::ERRORBAR_CATALOG_ENTRY, 2, Handles::Scalar),
        (&crate::BAR_CATALOG_ENTRY, 1, Handles::Series),
        (&crate::BARH_CATALOG_ENTRY, 1, Handles::Series),
        (&crate::AREA_CATALOG_ENTRY, 1, Handles::Series),
    ] {
        let mut request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); minimum],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, entry, minimum, handles);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        assert_eq!(
            result.outputs[0].shape,
            match handles {
                Handles::Scalar => ShapeFact::Scalar,
                Handles::Series => ShapeFact::Unknown,
            }
        );
        assert!(entry.contract.effects.contains(&EffectKind::UserInterface));
        request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!infer(&request, entry, minimum, handles)
            .diagnostics
            .is_empty());
        if minimum > 0 {
            request.arguments.clear();
            assert!(infer(&request, entry, minimum, handles)
                .diagnostics
                .iter()
                .any(|d| d.code == "RM-CATALOG-COORDINATE-ARITY"));
        }
    }
}

use runmat_runtime::builtins::common::spec::{canonical_fusion_declaration, FusionExprContext};

#[test]
fn unary_expression_declarations_never_discard_like_controls() {
    for name in ["single", "double", "sin", "cos", "tan", "atan"] {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name).unwrap();
        let [binding] = entry.bindings else {
            panic!("{name}: unique canonical binding");
        };
        let declaration = canonical_fusion_declaration(entry.binding_identity(binding)).unwrap();
        let template = declaration.spec.elementwise.as_ref().unwrap();
        assert!(!template.scalar_precisions.is_empty());
        for &scalar_ty in template.scalar_precisions {
            assert!(
                (template.wgsl_body)(&FusionExprContext {
                    scalar_ty,
                    inputs: &["x"],
                    constants: &[],
                })
                .is_ok(),
                "{name}: plain unary expression"
            );
            for inputs in [
                &[][..],
                &["x", "option"][..],
                &["x", "like", "prototype"][..],
            ] {
                assert!(
                    (template.wgsl_body)(&FusionExprContext {
                        scalar_ty,
                        inputs,
                        constants: &[],
                    })
                    .is_err(),
                    "{name}: {} operands must remain on the admitted runtime route",
                    inputs.len()
                );
            }
        }
    }
}

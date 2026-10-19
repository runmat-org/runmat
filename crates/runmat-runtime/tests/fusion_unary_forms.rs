use runmat_runtime::builtins::common::spec::{
    canonical_fusion_declaration, FusionExprContext, FusionExprInput,
};

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
                template
                    .wgsl_body
                    .render(&FusionExprContext {
                        scalar_ty,
                        inputs: &[FusionExprInput::text("x")],
                        constants: &[],
                    })
                    .is_ok(),
                "{name}: plain unary expression"
            );
            for inputs in [
                &[][..],
                &[FusionExprInput::text("x"), FusionExprInput::text("option")][..],
                &[
                    FusionExprInput::text("x"),
                    FusionExprInput::text("like"),
                    FusionExprInput::text("prototype"),
                ][..],
            ] {
                assert!(
                    template
                        .wgsl_body
                        .render(&FusionExprContext {
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

#[test]
fn selected_expression_writer_counts_the_bytes_it_emits() {
    for (name, inputs) in [
        (
            "plus",
            &[FusionExprInput::text("a"), FusionExprInput::text("b")][..],
        ),
        (
            "minus",
            &[FusionExprInput::text("a"), FusionExprInput::text("b")][..],
        ),
        (
            "times",
            &[FusionExprInput::text("a"), FusionExprInput::text("b")][..],
        ),
        (
            "rdivide",
            &[FusionExprInput::text("a"), FusionExprInput::text("b")][..],
        ),
        (
            "ldivide",
            &[FusionExprInput::text("a"), FusionExprInput::text("b")][..],
        ),
        (
            "power",
            &[FusionExprInput::text("a"), FusionExprInput::text("b")][..],
        ),
        ("sin", &[FusionExprInput::text("a")][..]),
        ("cos", &[FusionExprInput::text("a")][..]),
        ("tan", &[FusionExprInput::text("a")][..]),
        ("sinh", &[FusionExprInput::text("a")][..]),
        ("cosh", &[FusionExprInput::text("a")][..]),
        ("tanh", &[FusionExprInput::text("a")][..]),
        ("atan", &[FusionExprInput::text("a")][..]),
        ("asinh", &[FusionExprInput::text("a")][..]),
        ("acosh", &[FusionExprInput::text("a")][..]),
        ("atanh", &[FusionExprInput::text("a")][..]),
        ("exp", &[FusionExprInput::text("a")][..]),
        ("single", &[FusionExprInput::text("a")][..]),
        ("double", &[FusionExprInput::text("a")][..]),
    ] {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name).unwrap();
        let [binding] = entry.bindings else {
            panic!("{name}: unique canonical binding");
        };
        let declaration = canonical_fusion_declaration(entry.binding_identity(binding)).unwrap();
        let template = declaration.spec.elementwise.as_ref().unwrap();
        for &scalar_ty in template.scalar_precisions {
            let context = FusionExprContext {
                scalar_ty,
                inputs,
                constants: &[],
            };
            let measured = template.wgsl_body.measured_bytes(&context).unwrap();
            let rendered = template.wgsl_body.render(&context).unwrap();
            assert_eq!(measured, rendered.len(), "{name} {scalar_ty:?}");
            let lengths = inputs
                .iter()
                .map(|input| input.byte_len())
                .collect::<Vec<_>>();
            assert_eq!(
                template
                    .wgsl_body
                    .measured_bytes_from_lengths(scalar_ty, &lengths)
                    .unwrap(),
                rendered.len(),
                "{name} {scalar_ty:?}: virtual operand lengths"
            );
        }
    }
}

#[test]
fn registered_expression_writers_count_virtual_operands_exactly() {
    use runmat_runtime::builtins::common::spec::builtin_fusion_specs;

    let sources = [
        FusionExprInput::text("left"),
        FusionExprInput::text("right_operand"),
    ];
    let lengths = ["left".len(), "right_operand".len()];
    for spec in builtin_fusion_specs() {
        let Some(template) = &spec.elementwise else {
            continue;
        };
        for &scalar_ty in template.scalar_precisions {
            for arity in 0..=2 {
                let context = FusionExprContext {
                    scalar_ty,
                    inputs: &sources[..arity],
                    constants: &[],
                };
                let Ok(source) = template.wgsl_body.render(&context) else {
                    continue;
                };
                let counted = template
                    .wgsl_body
                    .measured_bytes_from_lengths(scalar_ty, &lengths[..arity])
                    .expect("admitted template has a measured form");
                assert_eq!(counted, source.len(), "{} {scalar_ty:?} {arity}", spec.name);
            }
        }
    }
}

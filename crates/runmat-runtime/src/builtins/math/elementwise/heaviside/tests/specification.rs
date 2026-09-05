use crate::builtins::common::spec::{FusionExprContext, ScalarType};
use runmat_builtins::MathInferenceRule;

#[test]
fn catalog_binding_and_acceleration_are_joined() {
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(super::super::BUILTIN_NAME);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        runmat_builtins::builtin_catalog_entry_by_name(super::super::BUILTIN_NAME)
            .expect("catalog entry")
            .contract
            .inference_rule,
        runmat_builtins::BuiltinInferenceRule::Math(MathInferenceRule::Heaviside)
    );
    assert_eq!(bindings[0].identity.variant, "default");
}

#[test]
fn fusion_expression_preserves_nan_and_zero() {
    let template = super::super::specification::FUSION_SPEC
        .elementwise
        .expect("fusion template");
    let inputs = ["x"];
    let expression = (template.wgsl_body)(&FusionExprContext {
        scalar_ty: ScalarType::F64,
        inputs: &inputs,
        constants: &[],
    })
    .expect("fusion expression");
    assert!(expression.contains("isNan(x)"));
    assert!(expression.contains("f64(0.5)"));
}

use super::*;
use runmat_types::NumericClass;

#[test]
fn explicit_host_single_input_proof_preserves_source_and_rejects_escapes() {
    let input =
        Value::Tensor(runmat_value::Tensor::new(vec![16_777_217.0, f64::NAN], vec![1, 2]).unwrap());
    let mut direct = plan(vec![cast("single", 0, 2)], 2);
    direct.inputs = vec![0];
    direct.stack_pattern = vec![0];
    let admitted = resolve_policy(
        &direct,
        std::slice::from_ref(&input),
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    let identity = admitted.cache_identity().unwrap();
    let proof = admitted.explicit_host_single_input(&direct, 0).unwrap();
    assert_eq!(
        proof.represented_class_for(&admitted, 0).unwrap(),
        NumericClass::Single
    );
    assert_eq!(admitted.floating_class(0).unwrap(), NumericClass::Double);
    let bound = direct
        .measured_typed_shader_preparation_bytes(&[2], &admitted, ScalarType::F32, 1)
        .unwrap();
    let shader = direct
        .prepare_typed_shader_with_inputs_bounded(
            &[2],
            &admitted,
            ScalarType::F32,
            &[0],
            Some(bound.body_bytes),
        )
        .unwrap();
    crate::wgsl::validation_tests::validate_generated(
        "explicit-host-single-input",
        shader.source(),
        64,
        8,
        crate::wgsl::validation_tests::RequiredCaps::None,
    );
    assert_eq!(identity, admitted.cache_identity().unwrap());
    assert!(admitted
        .explicit_host_single_input(&direct, 0)
        .unwrap()
        .represented_class_for(&admitted, 9)
        .is_err());
    let other_owner = resolve_policy(
        &direct,
        std::slice::from_ref(&input),
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    assert!(admitted
        .explicit_host_single_input(&direct, 0)
        .unwrap()
        .represented_class_for(&other_owner, 0)
        .is_err());

    // An original Double store escapes before the conversion and must retain
    // its source representation, even when the final output is Single.
    let mut escaped = direct.clone();
    escaped
        .materialized_stores
        .push(crate::fusion::FusionStoreMaterialization {
            value_id: 0,
            binding: crate::graph::VarBinding {
                kind: crate::graph::VarKind::Global,
                index: 0,
            },
        });
    assert!(admitted.explicit_host_single_input(&escaped, 0).is_none());
    for operations in [
        vec![
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![0, 0],
                output: Some(3),
            },
            cast("single", 3, 2),
        ],
        vec![
            cast("single", 0, 2),
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![0, 0],
                output: Some(3),
            },
        ],
    ] {
        let mut negative = plan(operations, 2);
        negative.inputs = vec![0];
        negative.stack_pattern = vec![0];
        let resolved = resolve_policy(
            &negative,
            std::slice::from_ref(&input),
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        )
        .unwrap();
        assert!(resolved.explicit_host_single_input(&negative, 0).is_none());
        assert!(negative
            .prepare_typed_shader(&[2], &resolved, ScalarType::F32)
            .is_err());
    }
}

use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn checked_double_add_emits_no_unused_ieee_helpers_or_int64_requirement() {
    let arithmetic = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    let program = resolve(&arithmetic, &[Value::Num(2.0), Value::Num(3.0)]).unwrap();
    let selected = arithmetic
        .prepare_typed_shader(&[2], &program, ScalarType::F64)
        .unwrap();
    assert_eq!(
        selected.requirements(),
        runmat_accelerate_api::ProviderShaderRequirements::Basic
    );
    assert!(!selected.source().contains("runmat_f64_"));
    assert!(!selected.source().contains("fn hypot("));
    crate::wgsl::validation_tests::validate_generated(
        "checked-double-add",
        selected.source(),
        64,
        8,
        crate::wgsl::validation_tests::RequiredCaps::Float64,
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn double_array_add_and_cast_keep_basic_shader_requirement() {
    let arithmetic = plan(
        vec![
            cast("double", 0, 2),
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![2, 1],
                output: Some(3),
            },
        ],
        3,
    );
    let inputs = [
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap()),
    ];
    let program = resolve(&arithmetic, &inputs).unwrap();
    let selected = arithmetic
        .prepare_typed_shader(&[3], &program, ScalarType::F64)
        .unwrap();
    assert_eq!(
        selected.requirements(),
        runmat_accelerate_api::ProviderShaderRequirements::Basic
    );
    assert!(!selected.source().contains("runmat_f64_"));
    crate::wgsl::validation_tests::validate_generated(
        "double-array-add-cast",
        selected.source(),
        64,
        8,
        crate::wgsl::validation_tests::RequiredCaps::Float64,
    );
}

#[test]
fn interior_single_arithmetic_remains_f32_before_double_publication() {
    let plan = plan(
        vec![
            cast("single", 0, 2),
            cast("single", 1, 3),
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![2, 3],
                output: Some(4),
            },
            cast("double", 4, 5),
        ],
        5,
    );
    let inputs = [Value::Num(16_777_216.0), Value::Num(1.0)];
    let program = resolve(&plan, &inputs).unwrap();
    assert_eq!(program.floating_class(4).unwrap(), NumericClass::Single);
    assert_eq!(program.floating_class(5).unwrap(), NumericClass::Double);
    let shader = plan
        .generate_typed_wgsl(&[5], &program, ScalarType::F64)
        .unwrap();
    assert!(shader.contains("let typed2: f32"), "{shader}");
    assert!(shader.contains("let typed3: f64"), "{shader}");
    // Host oracle demonstrates why widening only the final result is wrong.
    assert_eq!((16_777_216.0_f32 + 1.0_f32) as f64, 16_777_216.0);
    assert_eq!(16_777_216.0_f64 + 1.0_f64, 16_777_217.0);
}

#[test]
fn equal_final_class_does_not_collapse_different_interior_program_identities() {
    let rounded = plan(vec![cast("single", 0, 2), cast("double", 2, 3)], 3);
    let direct = plan(vec![cast("double", 0, 3)], 3);
    let inputs = [Value::Num(16_777_217.0), Value::Num(-0.0)];
    let first = resolve(&rounded, &inputs).unwrap();
    let second = resolve(&direct, &inputs).unwrap();
    assert_eq!(
        first.floating_class(3).unwrap(),
        second.floating_class(3).unwrap()
    );
    assert_ne!(
        first.cache_identity().unwrap(),
        second.cache_identity().unwrap()
    );
    assert_eq!(
        first.cache_identity().unwrap(),
        resolve(&rounded, &inputs)
            .unwrap()
            .cache_identity()
            .unwrap()
    );
}

#[test]
fn explicit_downcast_policy_changes_effective_facts_but_never_erases_double_conversion() {
    let arithmetic = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    let inputs = [Value::Num(16_777_216.0), Value::Num(1.0)];
    let exact = resolve_policy(
        &arithmetic,
        &inputs,
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    assert!(arithmetic
        .generate_typed_wgsl(&[2], &exact, ScalarType::F32)
        .is_err());
    let opted = resolve_policy(
        &arithmetic,
        &inputs,
        ProviderPrecision::F32,
        PrecisionPolicy::AllowDoubleToSingle,
    )
    .unwrap();
    assert_eq!(
        opted.source_inputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(opted.floating_class(2).unwrap(), NumericClass::Single);
    assert!(arithmetic
        .generate_typed_wgsl(&[2], &opted, ScalarType::F32)
        .is_ok());
    assert_ne!(
        exact.cache_identity().unwrap(),
        opted.cache_identity().unwrap()
    );
    let explicit = plan(vec![cast("double", 0, 2)], 2);
    let opted = resolve_policy(
        &explicit,
        &inputs,
        ProviderPrecision::F32,
        PrecisionPolicy::AllowDoubleToSingle,
    )
    .unwrap();
    assert_eq!(opted.floating_class(2).unwrap(), NumericClass::Double);
    assert!(explicit
        .generate_typed_wgsl(&[2], &opted, ScalarType::F32)
        .is_err());
}

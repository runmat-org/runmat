//! Source-to-physical projection witnesses; no allocation authority from WGSL.
use super::*;
use crate::fusion::reduction_recipe::{build, ReductionRecipeConfig};
use runmat_accelerate_api::scalar_program::{ScalarLiteral, ScalarNode};

fn minimum_plan() -> FusionGroupPlan {
    let mut source = plan();
    source
        .const_values
        .insert(5, Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()));
    let FusionOp::Builtin {
        name,
        origin,
        inputs,
        ..
    } = &mut source.operations[1]
    else {
        unreachable!()
    };
    *name = "min".into();
    *origin = FusionCallOrigin::named("min");
    *inputs = vec![2, 5, 4];
    source.reduction_flavor = None;
    source
}

#[test]
fn exact_product_words_survive_native_single_power_and_reduction_projection() {
    use runmat_accelerate_api::scalar_program::{ScalarImplementation, ScalarTransport};
    runmat_gc::gc_test_context(|| {
        for owner in [ProviderPrecision::F32, ProviderPrecision::F64] {
            for reverse in [false, true] {
                let mut source = if owner == ProviderPrecision::F32 {
                    minimum_plan()
                } else {
                    plan()
                };
                source.inputs.push(6);
                let FusionOp::Primitive { inputs, .. } = &mut source.operations[0] else {
                    unreachable!()
                };
                inputs[0] = 7;
                source.operations.insert(
                    0,
                    FusionOp::Primitive {
                        op: PrimitiveOp::Mul,
                        inputs: if reverse { vec![6, 8] } else { vec![8, 6] },
                        output: Some(7),
                    },
                );
                source.operations.insert(
                    0,
                    FusionOp::Primitive {
                        op: PrimitiveOp::Add,
                        inputs: vec![0, 0],
                        output: Some(8),
                    },
                );
                // Same-ID captured values cannot override the invocation.
                source.const_values.insert(6, Value::Num(0.1));
                let mut invocation = request(&source, Value::Num(2.0));
                invocation.inputs.push(Value::Num(4.0));
                let program = futures::executor::block_on(
                    crate::fusion_exec::resolve_reduction_program(&invocation, owner),
                )
                .unwrap()
                .unwrap();
                assert_eq!(program.producer_transport(), ScalarTransport::Binary32Words);
                assert_eq!(program.data_inputs().len(), 1);
                let implementations = program.producer_implementations();
                assert_eq!(
                    implementations.len(),
                    if owner == ProviderPrecision::F64 {
                        3
                    } else {
                        2
                    }
                );
                assert_eq!(implementations[0].output, 7);
                assert_eq!(
                    implementations[0].implementation,
                    ScalarImplementation::Binary32Product48RneV1
                );
                assert_eq!(implementations[0].invocations_per_element, 1);
                assert_eq!(
                    implementations[0].algorithmic_units_per_invocation,
                    crate::kernel_math::product_exact::work_prior().algorithmic_units()
                );
                if owner == ProviderPrecision::F64 {
                    assert_eq!(implementations[2].output, 2);
                    assert_eq!(
                        implementations[2].implementation,
                        ScalarImplementation::Binary32WidenToBinary64WordsV1
                    );
                }
                program.require_current_controls(&invocation).unwrap();
                let mut changed = request(&source, Value::Num(2.0));
                changed.inputs.push(Value::Num(8.0));
                assert!(program.require_current_controls(&changed).is_err());
                let mut changed_power = request(&source, Value::Num(1.0));
                changed_power.inputs.push(Value::Num(4.0));
                assert!(
                    program.require_current_controls(&changed_power).is_err(),
                    "all omitted controls, including POWER, retain original words"
                );
                let recipe = build(
                    &program,
                    ReductionRecipeConfig {
                        workgroup_size: 8,
                        chunk_elements: 1,
                        max_workgroup_size: 256,
                        max_workgroup_storage_bytes: 32768,
                        max_dispatch_dimension: 65535,
                        max_storage_binding_bytes: 1 << 30,
                        max_rank: 8,
                    },
                    None,
                )
                .unwrap();
                assert_eq!(recipe.physical().descriptor().partial.precision, owner);
                assert!(recipe.single_source().contains("data: array<u32>"));
                assert!(recipe.single_source().contains("bitcast<u32>(typed0)"));
                assert!(recipe
                    .single_source()
                    .contains("rm_exact_product_bits(typed0_bits, 0x40800000u)"));
                assert!(recipe
                    .single_source()
                    .contains("rm_exact_power_square_bits(typed1_bits)"));
                if owner == ProviderPrecision::F64 {
                    assert!(recipe
                        .single_source()
                        .contains("return runmat_f32_words_to_f64(typed2_bits)"));
                    assert_eq!(
                        recipe
                            .single_source()
                            .matches("fn runmat_f64_from_words(")
                            .count(),
                        1
                    );
                }
                crate::fusion::reduction_recipe::validate_lowered_sources(recipe.physical(), None)
                    .unwrap();
                let mut descriptor = recipe.physical().descriptor().clone();
                descriptor.producer_implementations.clear();
                let forged =
                    runmat_accelerate_api::reduction::ProviderReductionRecipe::new_checked(
                        recipe.physical().semantic_identity(),
                        descriptor,
                        recipe.physical().sources().clone(),
                        recipe.physical().checked_scalar().unwrap().clone(),
                        recipe.physical().scalar_inputs().to_vec(),
                    )
                    .unwrap();
                assert!(
                    crate::fusion::reduction_recipe::validate_lowered_sources(&forged, None)
                        .is_err()
                );
                #[cfg(feature = "wgpu")]
                for stage in [
                    recipe.single_source(),
                    recipe.pass1_source(),
                    recipe.pass2_source(),
                ] {
                    crate::wgsl::validation_tests::validate_generated(
                        "exact-product-reduction",
                        stage,
                        8,
                        8,
                        if owner == ProviderPrecision::F64 {
                            crate::wgsl::validation_tests::RequiredCaps::Float64Int64
                        } else {
                            crate::wgsl::validation_tests::RequiredCaps::None
                        },
                    );
                }
            }
        }
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn certified_controls_are_metadata_in_every_checked_reduction_stage() {
    runmat_gc::gc_test_context(|| {
        for owner in [ProviderPrecision::F32, ProviderPrecision::F64] {
            for exponent in [0.0_f64, -0.0, 1.0, 2.0] {
                for form in 0..3 {
                    let constant = form == 2;
                    let mut source = if owner == ProviderPrecision::F32 {
                        minimum_plan()
                    } else {
                        plan()
                    };
                    if constant {
                        source.inputs.pop();
                        source.const_values.insert(1, Value::Num(exponent));
                    }
                    let control = if form == 1 {
                        Value::Tensor(Tensor::new(vec![exponent], vec![1, 1]).unwrap())
                    } else {
                        Value::Num(exponent)
                    };
                    let mut invocation = request(&source, control);
                    if constant {
                        invocation.inputs.pop();
                    }
                    let program = futures::executor::block_on(
                        crate::fusion_exec::resolve_reduction_program(&invocation, owner),
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(program.data_inputs().len(), 1);
                    assert_eq!(program.data_inputs()[0].value_id, 0);
                    assert_eq!(program.execution_precision(), ProviderPrecision::F32);
                    let checked = program
                        .checked_scalar()
                        .expect("projected producer must be portable and checked");
                    assert!(checked.wire().nodes.iter().any(|node| matches!(node, ScalarNode::Literal { id: 1, value: ScalarLiteral::F64Bits(bits) } if *bits == exponent.to_bits())));
                    let recipe = build(
                        &program,
                        ReductionRecipeConfig {
                            workgroup_size: 8,
                            chunk_elements: 1,
                            max_workgroup_size: 256,
                            max_workgroup_storage_bytes: 32768,
                            max_dispatch_dimension: 65535,
                            max_storage_binding_bytes: 1 << 30,
                            max_rank: 8,
                        },
                        None,
                    )
                    .unwrap();
                    assert_eq!(recipe.physical().scalar_inputs(), &[0]);
                    assert_eq!(recipe.inputs().len(), 1);
                    assert_eq!(recipe.inputs()[0].bound_precision, ProviderPrecision::F32);
                    crate::fusion::reduction_recipe::validate_lowered_sources(
                        recipe.physical(),
                        None,
                    )
                    .unwrap();
                    assert!(recipe.single_source().contains("rm_exact_power_"));
                    assert!(recipe.single_source().contains("data: array<u32>"));
                    assert_eq!(recipe.physical().descriptor().partial.precision, owner);
                    #[cfg(feature = "wgpu")]
                    for stage in [
                        recipe.single_source(),
                        recipe.pass1_source(),
                        recipe.pass2_source(),
                    ] {
                        use crate::backend::wgpu::shaders::validation_tests::{
                            validate_generated, RequiredCaps,
                        };
                        validate_generated(
                            "certified-projected-reduction",
                            stage,
                            8,
                            2,
                            if owner == ProviderPrecision::F64 {
                                RequiredCaps::Float64Int64
                            } else {
                                RequiredCaps::None
                            },
                        );
                    }
                }
            }
        }
    });
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
async fn qualify_actual_projection(require_double_owner: bool, product: bool) {
    use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::{HostNumericDataOwned, HostNumericDataView, HostNumericTensorView};
    let provider = Box::leak(Box::new(
        WgpuProvider::new_async(WgpuProviderOptions::default())
            .await
            .expect("actual WGPU owner required"),
    ));
    // SAFETY: this actual facade and its unique allocation owner live forever.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    if require_double_owner {
        assert_eq!(
            provider.precision(),
            ProviderPrecision::F64,
            "real F64 owner is required, not substituted or skipped"
        );
    }
    let words = [1.25_f32, -2.0, f32::from_bits(1), -0.0];
    let product_source = |mut source: FusionGroupPlan| {
        if product {
            source.inputs.push(6);
            let FusionOp::Primitive { inputs, .. } = &mut source.operations[0] else {
                unreachable!()
            };
            inputs[0] = 7;
            source.operations.insert(
                0,
                FusionOp::Primitive {
                    op: PrimitiveOp::Mul,
                    inputs: vec![6, 0],
                    output: Some(7),
                },
            );
        }
        source
    };
    for (shape, transpose, reduce_len, slices, expected_shape) in [
        (vec![2, 2], false, 2, 2, vec![1, 2]),
        (vec![1, 4], false, 1, 4, vec![1, 4]),
        (vec![2, 2], true, 2, 2, vec![1, 2]),
    ] {
        let input = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&words),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let view = if transpose {
            provider.transpose(&input).unwrap()
        } else {
            input.clone()
        };
        // Integral but uncertified mixed control has static real-domain
        // evidence; rejection must not need a device probe, staging or submit.
        let negative_source = minimum_plan();
        let negative = FusionExecutionRequest {
            plan: &negative_source,
            inputs: vec![Value::GpuTensor(view.clone()), Value::Num(3.0)],
            placement: None,
            runtime: None,
        };
        let before_negative = provider.telemetry_snapshot();
        let rejected = execute_reduction_with_shape_placed_async(
            negative,
            reduce_len,
            slices,
            &expected_shape,
            0,
        )
        .await;
        assert!(
            rejected.is_err() || matches!(rejected, Ok(FusionExecutionOutcome::SharedRuntime(_)))
        );
        assert_eq!(
            provider.telemetry_snapshot(),
            before_negative,
            "unsupported mixed family must make zero provider effects"
        );
        for exponent in [0.0_f64, 1.0, 2.0] {
            for synchronous in [false, true] {
                let source = product_source(minimum_plan());
                let mut invocation = FusionExecutionRequest {
                    plan: &source,
                    inputs: vec![Value::GpuTensor(view.clone()), Value::Num(exponent)],
                    placement: None,
                    runtime: None,
                };
                if product {
                    invocation.inputs.push(Value::Num(4.0));
                }
                let before = provider.telemetry_snapshot();
                let output = if synchronous {
                    execute_reduction_with_shape_placed(
                        invocation,
                        reduce_len,
                        slices,
                        &expected_shape,
                        0,
                    )
                } else {
                    execute_reduction_with_shape_placed_async(
                        invocation,
                        reduce_len,
                        slices,
                        &expected_shape,
                        0,
                    )
                    .await
                }
                .unwrap();
                let FusionExecutionOutcome::Executed(Value::GpuTensor(output)) = output else {
                    panic!("actual projected reduction must execute")
                };
                let after = provider.telemetry_snapshot();
                assert_eq!(
                    after.upload_bytes, before.upload_bytes,
                    "metadata control must not upload"
                );
                assert_eq!(
                    after.download_bytes, before.download_bytes,
                    "admission must not gather"
                );
                assert_eq!(output.shape(), expected_shape);
                assert_eq!(output.physical().element_type(), NumericElementType::F32);
                let downloaded = provider.download_numeric(&output).await.unwrap();
                let HostNumericDataOwned::F32(actual) = downloaded.data else {
                    panic!("Single publication required")
                };
                let logical = if transpose {
                    [words[0], words[2], words[1], words[3]]
                } else {
                    words
                };
                let transformed = logical.map(|base| {
                    let base = if product {
                        (f64::from(base) * 4.0) as f32
                    } else {
                        base
                    };
                    (f64::from(base).powf(exponent)) as f32
                });
                if reduce_len == 1 {
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        transformed.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                } else {
                    assert_eq!(
                        actual[0].to_bits(),
                        transformed[0].min(transformed[1]).to_bits()
                    );
                }
                output.release().unwrap();
            }
        }
        drop(view);
        input.release().unwrap();
    }
    if require_double_owner {
        // Genuine Double accumulation is retained while all producer loads
        // remain original binary32. This is not an f32 accumulation substitute.
        let source = product_source(plan());
        let mut invocation = FusionExecutionRequest {
            plan: &source,
            inputs: vec![
                Value::Tensor(Tensor::from_f32(vec![1.25, -2.0], vec![2, 1]).unwrap()),
                Value::Num(2.0),
            ],
            placement: None,
            runtime: None,
        };
        if product {
            invocation.inputs.push(Value::Num(4.0));
        }
        let output = execute_reduction_with_shape_placed_async(invocation, 2, 1, &[1, 1], 0)
            .await
            .unwrap();
        let FusionExecutionOutcome::Executed(Value::GpuTensor(output)) = output else {
            panic!("real F64 accumulator route required")
        };
        let result = provider.download_numeric(&output).await.unwrap();
        assert_eq!(
            result.data,
            HostNumericDataOwned::F32(vec![if product { 89.0 } else { 5.5625 }])
        );
        output.release().unwrap();
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[tokio::test]
#[ignore = "requires actual WGPU adapter; qualifies authenticated mixed POWER reduction projection"]
async fn actual_certified_power_reduction_uses_projected_binary32_data() {
    qualify_actual_projection(false, false).await;
    qualify_actual_projection(false, true).await;
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[tokio::test]
#[ignore = "requires actual F64 WGPU owner; never substitutes F32 hardware"]
async fn actual_f64_owner_power_reduction_keeps_binary32_data() {
    qualify_actual_projection(true, false).await;
    qualify_actual_projection(true, true).await;
}

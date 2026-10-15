//! Shared request-owner tests; no image-specific representation authority.
use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorLayout, GpuTensorStorage, HostTensorView, NumericElementType, TransposeInfo,
};
use runmat_types::StorageFact;

struct ResidentProvider(TestAllocationOwner);
impl AccelProvider for ResidentProvider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "resident carrier fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("carrier certification must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("carrier certification must not gather")
    }
}

fn owner() -> &'static ResidentProvider {
    let provider = Box::leak(Box::new(ResidentProvider(TestAllocationOwner::new())));
    // SAFETY: leaked lifetime and privately minted distinct fixture device.
    // No ambient provider selection is changed.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn exact_host_resources_keep_double_intermediates_beside_automatic_single_inputs() {
    use runmat_runtime::builtins::common::spec::{
        FusionHostResourceInput, FusionHostResourceQuery,
    };

    let provider = owner();
    let resident = Value::GpuTensor(
        provider
            .0
            .allocate_zeroed(vec![1, 3], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap()
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap(),
    );
    let double =
        Value::Tensor(runmat_value::Tensor::new(vec![0.25, 0.5, 0.75], vec![1, 3]).unwrap());
    let mut source = plan(
        vec![
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![1, 4],
                output: Some(2),
            },
            cast("exp", 2, 3),
            FusionOp::Builtin {
                origin: Default::default(),
                name: "times".into(),
                inputs: vec![3, 0],
                output: Some(5),
            },
        ],
        5,
    );
    source.const_values.insert(4, Value::Num(1.0));
    let bindings = semantic_bindings::capture(&source).unwrap();
    let host = ResolvedProgram::resolve_host_static_captured(
        &source,
        &[&resident, &double],
        ProviderPrecision::F32,
        &bindings,
    )
    .unwrap()
    .unwrap();
    let exp_bound = |program: &ResolvedProgram| {
        bindings[1].unwrap().declaration.host_resources.unwrap()(&FusionHostResourceQuery {
            output_domain: runmat_runtime::builtins::common::spec::FusionHostOutputDomain::Exact(
                runmat_types::NumericDomain::Real,
            ),
            inputs: &[FusionHostResourceInput {
                shape: &[1, 3],
                class: program.floating_class(2).unwrap(),
            }],
            output: FusionHostResourceInput {
                shape: &[1, 3],
                class: program.floating_class(3).unwrap(),
            },
        })
        .unwrap()
    };
    for policy in [PrecisionPolicy::Exact, PrecisionPolicy::AllowDoubleToSingle] {
        let effective = resolve_policy(
            &source,
            &[resident.clone(), double.clone()],
            ProviderPrecision::F32,
            policy,
        )
        .unwrap();
        assert_eq!(host.floating_class(0).unwrap(), NumericClass::Single);
        for id in [1, 2, 3, 4] {
            assert_eq!(host.floating_class(id).unwrap(), NumericClass::Double);
        }
        assert_eq!(host.floating_class(5).unwrap(), NumericClass::Single);
        let actual = exp_bound(&host);
        assert_eq!(
            actual.retained_bytes,
            3 * 8 + 2 * std::mem::size_of::<usize>() as u64
        );
        let projected = exp_bound(&effective);
        if policy == PrecisionPolicy::AllowDoubleToSingle {
            assert_eq!(effective.floating_class(3).unwrap(), NumericClass::Single);
            assert_eq!(actual.retained_bytes - projected.retained_bytes, 3 * 4);
            assert_eq!(actual.scratch_bytes - projected.scratch_bytes, 3 * 4);
        } else {
            assert_eq!(actual.retained_bytes, projected.retained_bytes);
        }
    }
}

fn storage(leaves: &FusionInvocationLeaves<'_>, id: ValueId, value: &Value) -> ValueFact {
    let mut fact = value_fact(value);
    leaves.refine_input_storage(id, value, &mut fact);
    fact
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn resident_carrier_is_request_bound_and_preserves_public_alias_facets() {
    let provider = owner();
    let allocation = provider
        .0
        .allocate_zeroed(
            vec![2, 4, 5],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let input = Value::GpuTensor(allocation.handle.clone());
    let epsilon = Value::Num(1.0);
    let source = plan(vec![cast("single", 0, 2)], 2);
    let leaves = FusionInvocationLeaves::new(&source, &[&input, &epsilon]).unwrap();
    let public = value_fact(&input);
    assert_eq!(public.storage, StorageFact::Opaque);
    let mut expected = public.clone();
    expected.storage = StorageFact::Dense;
    assert_eq!(storage(&leaves, 0, &input), expected);
    assert_eq!(value_fact(&input), public);
    assert_eq!(storage(&leaves, 99, &input), public);
    let replacement = provider
        .0
        .allocate_zeroed(
            vec![2, 4, 5],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let replacement = Value::GpuTensor(replacement.handle);
    assert_eq!(storage(&leaves, 0, &replacement), value_fact(&replacement));
    let sibling = Value::GpuTensor(
        allocation
            .handle
            .with_view(allocation.handle.view().clone())
            .unwrap(),
    );
    assert_eq!(storage(&leaves, 0, &sibling), value_fact(&sibling));
    let replacement = provider
        .0
        .allocate_zeroed(
            vec![2, 3, 4],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let replacement = Value::GpuTensor(replacement.handle);
    assert_eq!(storage(&leaves, 0, &replacement), value_fact(&replacement));
    let replacement = provider
        .0
        .allocate_zeroed(
            vec![2, 4, 5],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let replacement = Value::GpuTensor(replacement.handle);
    assert_eq!(storage(&leaves, 0, &replacement), value_fact(&replacement));
    allocation.probe.mark_unavailable();
    assert_eq!(storage(&leaves, 0, &input), public);
    assert!(leaves.require_live_resident_inputs().is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn resident_carrier_rejects_unowned_transposed_logical_and_nonreal_inputs() {
    let provider = owner();
    let source = plan(vec![cast("single", 0, 2)], 2);
    let epsilon = Value::Num(1.0);
    let dense = provider
        .0
        .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let transpose = dense
        .handle
        .with_view(
            dense
                .handle
                .view()
                .clone()
                .with_shape(vec![3, 2])
                .with_layout(GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 3,
                })),
        )
        .unwrap();
    let unregistered = TestAllocationOwner::new()
        .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let logical = provider
        .0
        .allocate_logical_zeroed(vec![2, 3], ProviderPrecision::F32)
        .unwrap();
    let complex = provider
        .0
        .allocate_zeroed(
            vec![2, 3],
            NumericElementType::F32,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    let integer = provider
        .0
        .allocate_zeroed(vec![2, 3], NumericElementType::I32, GpuTensorStorage::Real)
        .unwrap();
    for handle in [
        transpose,
        unregistered.handle,
        logical.handle,
        complex.handle,
        integer.handle,
    ] {
        let input = Value::GpuTensor(handle);
        let leaves = FusionInvocationLeaves::new(&source, &[&input, &epsilon]).unwrap();
        assert_eq!(storage(&leaves, 0, &input), value_fact(&input));
    }
    let dead = provider
        .0
        .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    dead.probe.mark_unavailable();
    let input = Value::GpuTensor(dead.handle);
    let leaves = FusionInvocationLeaves::new(&source, &[&input, &epsilon]).unwrap();
    assert_eq!(storage(&leaves, 0, &input), value_fact(&input));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn static_and_async_resident_input_facts_share_identity_without_opted_class_proof() {
    let provider = owner();
    let input = Value::GpuTensor(
        provider
            .0
            .allocate_zeroed(
                vec![2, 4, 5],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let epsilon = Value::Num(1.0);
    let source = plan(vec![cast("single", 0, 2)], 2);
    let bindings = semantic_bindings::capture(&source).unwrap();
    let static_program = ResolvedProgram::resolve_static_captured(
        &source,
        &[&input, &epsilon],
        ProviderPrecision::F64,
        &bindings,
    )
    .unwrap()
    .unwrap();
    let async_program = futures::executor::block_on(ResolvedProgram::resolve_captured(
        &source,
        &[&input, &epsilon],
        ProviderPrecision::F64,
        None,
        &bindings,
    ))
    .unwrap();
    assert_eq!(
        static_program.cache_identity().unwrap(),
        async_program.cache_identity().unwrap()
    );
    assert_eq!(async_program.fact(0).unwrap().storage, StorageFact::Dense);
    assert_eq!(async_program.source_inputs[0], value_fact(&input));
    let (opted, _) = source_assembly(
        &source,
        &[&input, &epsilon],
        ProviderPrecision::F32,
        PrecisionPolicy::AllowDoubleToSingle,
        None,
        &bindings,
    )
    .unwrap();
    assert_eq!(
        opted.fact(0).unwrap().numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        opted.source_inputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(opted.source_inputs[0].storage, StorageFact::Opaque);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn canonical_mean_owner_certifies_publication_only_after_resident_carrier() {
    use runmat_runtime::builtins::common::spec::FusionReductionContext;
    let provider = owner();
    let input = Value::GpuTensor(
        provider
            .0
            .allocate_zeroed(
                vec![2, 4, 5],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let epsilon = Value::Num(1.0);
    let mut source = plan(
        vec![FusionOp::Builtin {
            origin: Default::default(),
            name: "mean".into(),
            inputs: vec![0, 3],
            output: Some(2),
        }],
        2,
    );
    source.const_values.insert(
        3,
        Value::Tensor(runmat_value::Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
    );
    let program = resolve(&source, &[input.clone(), epsilon]).unwrap();
    let bindings = semantic_bindings::capture(&source).unwrap();
    let declaration = bindings[0].unwrap().declaration.reduction.unwrap();
    let controls = [source.const_values[&3].clone()];
    let output = program.fact(2).unwrap();
    assert_eq!(output.storage, StorageFact::Unknown);
    let certify = |fact: &ValueFact| {
        futures::executor::block_on(declaration.certify_publication_storage(
            FusionReductionContext {
                input: fact,
                controls: &controls,
                requested_outputs: 1,
            },
            output,
        ))
        .unwrap()
    };
    assert_eq!(certify(&value_fact(&input)), None);
    assert_eq!(certify(program.fact(0).unwrap()), Some(StorageFact::Dense));
    assert_eq!(program.fact(2).unwrap().storage, StorageFact::Unknown);
}

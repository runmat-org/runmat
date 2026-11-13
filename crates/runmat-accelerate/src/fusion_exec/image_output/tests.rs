use super::*;
use runmat_accelerate_api::{
    test_support::{TestAllocation, TestAllocationOwner},
    AccelDownloadFuture, GpuTensorLayout, GpuTensorView, HostTensorView, ProviderPrecision,
    ThreadProviderGuard, TransposeInfo,
};

struct Owner {
    allocations: TestAllocationOwner,
    claimed_device: Option<u32>,
}

fn owner() -> &'static Owner {
    Box::leak(Box::new(Owner {
        allocations: TestAllocationOwner::new(),
        claimed_device: None,
    }))
}

impl AccelProvider for Owner {
    fn device_id(&self) -> u32 {
        self.claimed_device
            .unwrap_or_else(|| self.allocations.device_id())
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "image-output-fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("image output admission never uploads")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!("image output admission never downloads")
    }
}

fn numeric(owner: &Owner, element: NumericElementType) -> TestAllocation {
    shaped_numeric(owner, vec![1, 2], element)
}

fn shaped_numeric(owner: &Owner, shape: Vec<usize>, element: NumericElementType) -> TestAllocation {
    owner
        .allocations
        .allocate_zeroed(shape, element, GpuTensorStorage::Real)
        .expect("owned numeric allocation")
}

fn reject_and_keep_input(owner: &'static Owner, input: &TestAllocation, output: TestAllocation) {
    let probe = output.probe.clone();
    assert!(admit(output.handle, owner, &input.handle).is_err());
    assert_eq!(probe.retirements(), 1, "only the invalid output retires");
    assert_eq!(input.probe.retirements(), 0, "input must remain live");
    input
        .probe
        .with_data(&input.handle, |view| assert_eq!(view.len(), 2))
        .expect("input remains readable");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn image_output_admits_exact_single_and_double_without_aliasing_input() {
    let owner = owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    for element in [NumericElementType::F32, NumericElementType::F64] {
        let input = numeric(owner, element);
        let output = numeric(owner, element);
        let admitted = admit(output.handle, owner, &input.handle).expect("valid output");
        assert_eq!(admitted.shape(), input.handle.shape());
        assert_eq!(admitted.physical().element_type(), element);
        assert!(!admitted.shares_allocation_with(&input.handle));
        assert_eq!(input.probe.retirements(), 0);
        assert_eq!(output.probe.retirements(), 0);
        drop(admitted);
        assert_eq!(output.probe.retirements(), 1);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn image_output_rejects_actual_wrong_class_storage_logical_shape_and_layout() {
    let owner = owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let input = numeric(owner, NumericElementType::F64);
    reject_and_keep_input(owner, &input, numeric(owner, NumericElementType::F32));
    reject_and_keep_input(
        owner,
        &input,
        owner
            .allocations
            .allocate_zeroed(
                vec![1, 2],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .expect("complex storage"),
    );
    reject_and_keep_input(
        owner,
        &input,
        owner
            .allocations
            .allocate_logical_zeroed(vec![1, 2], ProviderPrecision::F64)
            .expect("logical allocation"),
    );
    reject_and_keep_input(
        owner,
        &input,
        owner
            .allocations
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .expect("different shape"),
    );
    let transposed = numeric(owner, NumericElementType::F64);
    let view =
        transposed
            .handle
            .with_view(GpuTensorView::contiguous(vec![1, 2]).with_layout(
                GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 1,
                }),
            ))
            .expect("legal transposed view");
    drop(transposed.handle);
    reject_and_keep_input(
        owner,
        &input,
        TestAllocation {
            handle: view,
            probe: transposed.probe,
        },
    );
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn image_output_rejects_input_alias_and_unavailable_output_without_retiring_input() {
    let owner = owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let input = numeric(owner, NumericElementType::F64);
    assert!(admit(input.handle.clone(), owner, &input.handle).is_err());
    assert_eq!(input.probe.retirements(), 0);
    let unavailable = numeric(owner, NumericElementType::F64);
    unavailable.probe.mark_unavailable();
    reject_and_keep_input(owner, &input, unavailable);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn image_output_rejects_foreign_registered_owner_even_with_claimed_device_id() {
    let registered = owner();
    // A distinct facade may report the same integer device ID, but only the
    // registered owner is authorized to resolve this allocation's namespace.
    unsafe { runmat_accelerate_api::register_device_provider(registered) };
    let foreign = Box::leak(Box::new(Owner {
        allocations: TestAllocationOwner::new(),
        claimed_device: Some(registered.device_id()),
    }));
    let _selected = ThreadProviderGuard::set(Some(foreign));
    let input = numeric(registered, NumericElementType::F64);
    let output = numeric(registered, NumericElementType::F64);
    let probe = output.probe.clone();
    assert!(admit(output.handle, foreign, &input.handle).is_err());
    assert_eq!(probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn staged_moment_requires_exact_geometry_class_owner_and_live_lease() {
    let selected = owner();
    let _selected = ThreadProviderGuard::set(Some(selected));
    let input = shaped_numeric(selected, vec![2, 3, 4], NumericElementType::F32);
    let expected = [2, 1, 1];
    for output in [
        shaped_numeric(selected, vec![2, 3, 4], NumericElementType::F32),
        shaped_numeric(selected, expected.to_vec(), NumericElementType::F64),
    ] {
        let probe = output.probe.clone();
        let error = admit_stage(output.handle, selected, &input.handle, &expected, &[])
            .expect_err("malformed moment cannot be published");
        assert!(!error.is_unsupported(), "allocated result forbids replay");
        assert_eq!(probe.retirements(), 1);
    }
    let unavailable = shaped_numeric(selected, expected.to_vec(), NumericElementType::F32);
    unavailable.probe.mark_unavailable();
    let probe = unavailable.probe.clone();
    assert!(admit_stage(unavailable.handle, selected, &input.handle, &expected, &[]).is_err());
    assert_eq!(probe.retirements(), 1);

    let foreign = owner();
    let output = shaped_numeric(foreign, expected.to_vec(), NumericElementType::F32);
    let probe = output.probe.clone();
    assert!(admit_stage(output.handle, selected, &input.handle, &expected, &[]).is_err());
    assert_eq!(probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);

    let mean = shaped_numeric(selected, expected.to_vec(), NumericElementType::F32);
    let mean_probe = mean.probe.clone();
    let mean = admit_stage(mean.handle, selected, &input.handle, &expected, &[])
        .expect("exact moment admitted");
    let sigma = shaped_numeric(selected, expected.to_vec(), NumericElementType::F32);
    let sigma_probe = sigma.probe.clone();
    let sigma = admit_stage(sigma.handle, selected, &input.handle, &expected, &[&mean])
        .expect("distinct sigma admitted");
    assert!(!mean.shares_allocation_with(&sigma));
    assert_eq!(mean_probe.retirements(), 0);
    assert_eq!(sigma_probe.retirements(), 0);
    drop(mean);
    drop(sigma);
    assert_eq!(mean_probe.retirements(), 1);
    assert_eq!(sigma_probe.retirements(), 1);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn staged_output_rejects_aliases_without_retiring_published_or_borrowed_handles() {
    let selected = owner();
    let _selected = ThreadProviderGuard::set(Some(selected));
    let input = shaped_numeric(selected, vec![2, 1, 1], NumericElementType::F32);
    let published = shaped_numeric(selected, vec![2, 1, 1], NumericElementType::F32);
    let published_probe = published.probe.clone();
    let borrowed_probe = input.probe.clone();
    let expected = [2, 1, 1];
    let error = admit_stage(
        input.handle.clone(),
        selected,
        &input.handle,
        &expected,
        &[&published.handle],
    )
    .expect_err("input alias cannot be published");
    assert!(!error.is_unsupported());
    let error = admit_stage(
        published.handle.clone(),
        selected,
        &input.handle,
        &expected,
        &[&published.handle],
    )
    .expect_err("prior output alias cannot be republished");
    assert!(!error.is_unsupported());
    assert_eq!(borrowed_probe.retirements(), 0);
    assert_eq!(published_probe.retirements(), 0);
    borrowed_probe
        .with_data(&input.handle, |data| assert_eq!(data.len(), 2))
        .expect("borrowed input remains readable");
    published_probe
        .with_data(&published.handle, |data| assert_eq!(data.len(), 2))
        .expect("published output remains readable");
    drop(input.handle);
    drop(published.handle);
    assert_eq!(borrowed_probe.retirements(), 1);
    assert_eq!(published_probe.retirements(), 1);
}

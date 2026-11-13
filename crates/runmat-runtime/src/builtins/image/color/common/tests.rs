use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{IntValue, IntegerStorage};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn color_geometry_preserves_full_rank_and_rejects_non_image_ranks() {
    let stacked = Tensor::new(vec![0.0; 12], vec![2, 1, 3, 2]).unwrap();
    let layout = stacked_color_layout(&stacked, "color-test").unwrap();
    assert_eq!(layout.pixels(), 4);
    assert_eq!(layout.output_shape(), vec![2, 1, 3, 2]);

    let rank_zero = Tensor::new(vec![0.0], Vec::new()).unwrap();
    assert!(color_layout(&rank_zero, "color-test").is_err());
    assert!(grayscale_shape(&rank_zero, "color-test").is_err());

    let rank_one = Tensor::new(vec![0.0; 3], vec![3]).unwrap();
    assert!(color_layout(&rank_one, "color-test").is_err());
    assert!(grayscale_shape(&rank_one, "color-test").is_err());
}

struct PublicationOwner(runmat_accelerate_api::test_support::TestAllocationOwner);

impl runmat_accelerate_api::AccelProvider for PublicationOwner {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }

    fn device_info(&self) -> String {
        "color publication lease witness".into()
    }

    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unavailable source must reject before upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("publication must not download again")
    }
}

#[test]
fn publication_revalidates_a_source_that_became_unavailable_after_readback() {
    let _state = test_support::accel_test_lock();
    let provider: &'static PublicationOwner = Box::leak(Box::new(PublicationOwner(
        runmat_accelerate_api::test_support::TestAllocationOwner::new(),
    )));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let first = provider
        .0
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let second = provider
        .0
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let sources = [first.handle, second.handle];
    for source in &sources {
        super::super::resident_owner::validate_live(source, "color-test").unwrap();
    }
    // Independent native payload has already returned; provider loss makes
    // only the second borrowed source unavailable before result publication.
    let host_result = Value::Tensor(Tensor::new(vec![0.25; 3], vec![1, 1, 3]).unwrap());
    second.probe.mark_unavailable();
    let error = restore_resident_numeric_result_for_sources(&sources, host_result, "color-test")
        .expect_err("lost readback source cannot authorize publication");
    assert!(error.message().contains("invalid resident allocation"));
    assert!(sources[0].allocation_lease().validate_live().is_ok());
    assert_eq!(first.probe.retirements(), 0);
    assert_eq!(second.probe.retirements(), 0);
}

#[test]
fn image_scalar_collapse_reads_authoritative_storage() {
    let single = Tensor::from_f32(vec![0.25], vec![1, 1]).unwrap();
    assert_eq!(image_value_from_tensor(single), Value::Num(0.25));

    let wide = u64::MAX - 1;
    let integer = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
    assert_eq!(
        image_value_from_tensor(integer),
        Value::Int(IntValue::U64(wide))
    );
}

#[test]
fn image_arrays_remain_tensors() {
    let tensor = Tensor::from_f32(vec![0.25, 0.5], vec![1, 2]).unwrap();
    let Value::Tensor(output) = image_value_from_tensor(tensor) else {
        panic!("expected tensor output");
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), vec![0.25, 0.5]);
}

#[test]
fn rejects_legacy_global_provider_fallback_for_foreign_device_handle() {
    test_support::with_test_provider(|provider| {
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F32 => {
                        runmat_accelerate_api::NumericElementType::F32
                    }
                    runmat_accelerate_api::ProviderPrecision::F64 => {
                        runmat_accelerate_api::NumericElementType::F64
                    }
                },
                GpuTensorStorage::Real,
            )
            .expect("admit owned fixture")
            .handle;
        let error = resident_numeric_dtype(&handle, "color-test")
            .expect_err("global fallback is not exact ownership");
        assert!(error.message().contains("not owned"));
    });
}

#[test]
fn native_single_restore_does_not_depend_on_preferred_double_precision() {
    test_support::with_test_provider(|provider| {
        let source_tensor = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &source_tensor).unwrap();
        let single = Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap());

        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("publish provenance view");
        assert!(matches!(
            restore_resident_numeric_result(Some(&source), single.clone(), "color-test").unwrap(),
            Value::GpuTensor(_)
        ));

        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let result = restore_resident_numeric_result(Some(&source), single, "color-test")
            .expect("owner supports actual native Single storage");
        let Value::GpuTensor(output) = result else {
            panic!("resident output")
        };
        assert_eq!(
            output.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F32
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        source.release().unwrap();
    });
}

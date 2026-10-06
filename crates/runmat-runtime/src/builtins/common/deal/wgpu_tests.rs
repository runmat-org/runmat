use super::*;
use runmat_accelerate_api::{HostIntegerDataOwned, IntegerElementType};
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn deal_wgpu_resident_integer_aliases_when_a_device_is_available() {
    check_wgpu_resident_aliases(false);
}

#[test]
#[ignore = "requires a compatible WGPU device; run explicitly for hardware qualification"]
fn deal_required_wgpu_resident_integer_aliases() {
    check_wgpu_resident_aliases(true);
}

fn check_wgpu_resident_aliases(require_device: bool) {
    let _provider_guard = crate::builtins::common::test_support::accel_test_lock();
    let Some(provider) = crate::builtins::common::test_support::wgpu_provider_if_available() else {
        assert!(!require_device, "required WGPU device is unavailable");
        return;
    };
    runmat_gc::gc_test_context(|| {
        let original = vec![9_007_199_254_740_993, u64::MAX];
        let tensor = Tensor::new_integer(IntegerStorage::U64(original.clone()), vec![1, 2])
            .expect("wide integer input");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("real WGPU resident upload");
        let lease = handle.allocation_lease().downgrade();
        let weak_view = handle.downgrade();
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&handle),
            Some(IntegerElementType::U64)
        );
        let nested = Value::Cell(
            runmat_value::CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1)
                .expect("nested resident input"),
        );
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let _outputs = crate::output_count::push_output_count(Some(2));
            let error = futures::executor::block_on(deal_builtin_registered(vec![nested.clone()]))
                .expect_err("resident extension must be explicit");
            assert_eq!(
                error.identifier(),
                DEAL_RESIDENT_INPUT_EXTENSION.error_identifier
            );
        }
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _outputs = crate::output_count::push_output_count(Some(2));
        let transfers_before = provider.telemetry_snapshot();
        let result = futures::executor::block_on(deal_builtin_registered(vec![nested.clone()]))
            .expect("resident aliases");
        assert_eq!(
            result,
            Value::OutputList(vec![nested.clone(), nested.clone()])
        );
        let Value::OutputList(outputs) = &result else {
            panic!("two resident outputs");
        };
        for output in outputs {
            let Value::Cell(cell) = output else {
                panic!("nested resident output");
            };
            let Value::GpuTensor(alias) = &cell.data[0] else {
                panic!("resident alias");
            };
            assert!(alias.shares_allocation_with(&handle));
            assert_eq!(alias.allocation_identity(), handle.allocation_identity());
            assert!(runmat_accelerate_api::same_provider_instance(
                runmat_accelerate_api::provider_for_handle(alias).expect("output owner"),
                provider
            ));
        }
        let transfers_after = provider.telemetry_snapshot();
        assert_eq!(transfers_after.upload_bytes, transfers_before.upload_bytes);
        assert_eq!(
            transfers_after.download_bytes,
            transfers_before.download_bytes
        );
        let output_roots =
            crate::execution::RootedValueHandles::for_values(std::iter::once(&result))
                .expect("root returned aliases during collection");
        drop(nested);
        drop(handle);
        runmat_gc::gc_collect_major().expect("collect with live returned aliases");
        let output_handle = weak_view
            .acquire()
            .expect("returned aliases retain the view");
        let downloaded = futures::executor::block_on(provider.download_integer(&output_handle))
            .expect("download exact WGPU input after deal");
        assert_eq!(downloaded.data, HostIntegerDataOwned::U64(original));
        drop(output_handle);
        drop(output_roots);
        drop(result);
        runmat_gc::gc_collect_major().expect("collect after returned aliases retire");
        assert!(
            lease.acquire().is_none(),
            "returned aliases must retire their owner"
        );
        assert!(
            weak_view.acquire().is_err(),
            "returned aliases must retire their view"
        );
    });
}

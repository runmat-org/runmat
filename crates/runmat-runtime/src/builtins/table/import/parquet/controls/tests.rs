use super::*;

fn run_parquetread(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    block_on(crate::builtins::table::builtins::parquetread_builtin(
        None, path, rest,
    ))
}
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned,
    HostTensorView, ProviderPrecision,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct ControlOwner {
    allocations: TestAllocationOwner,
    downloads: AtomicUsize,
}
impl AccelProvider for ControlOwner {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn device_info(&self) -> String {
        "parquet-control-owner".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("no upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("preserve native control class")
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            handle.allocation_lease().validate_live()?;
            self.downloads.fetch_add(1, Ordering::SeqCst);
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(vec![2.0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn controls_admit_only_needed_readback_after_host_validation() {
    runmat_gc::gc_test_context(|| {
        let owner = Box::leak(Box::new(ControlOwner {
            allocations: TestAllocationOwner::new(),
            downloads: AtomicUsize::new(0),
        }));
        // SAFETY: this unique device owner remains alive for the test process.
        unsafe { runmat_accelerate_api::register_device_provider(owner) };
        let allocation = owner
            .allocations
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(vec![2.0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let resident = Value::GpuTensor(allocation.handle.clone());
        {
            let _outputs =
                crate::output_count::push_output_count(Some(2)).expect("output-count scope");
            let errors = [
                block_on(crate::builtins::table::builtins::parquetinfo_builtin(
                    Value::from("not-read.parquet"),
                ))
                .unwrap_err(),
                run_parquetread(
                    Value::from("not-read.parquet"),
                    vec![Value::from("RowGroups"), resident.clone()],
                )
                .unwrap_err(),
            ];
            for error in errors {
                assert_eq!(error.identifier(), TABLE_ERROR_INVALID_ARGUMENT.identifier);
                assert_eq!(error.message(), TABLE_ERROR_INVALID_ARGUMENT.message);
            }
            assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        }
        assert!(
            block_on(crate::builtins::table::builtins::parquetinfo_builtin(
                resident.clone()
            ))
            .is_err()
        );
        assert!(run_parquetread(resident.clone(), vec![]).is_err());
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        for args in [
            vec![resident.clone(), Value::Num(2.0)],
            vec![Value::from("OutputType"), resident.clone()],
            vec![
                Value::from("RowGroups"),
                resident.clone(),
                Value::from("unknown"),
                Value::Num(1.0),
            ],
            vec![
                Value::from("RowGroups"),
                resident.clone(),
                Value::from("OutputType"),
                Value::from("timetable"),
            ],
        ] {
            assert!(block_on(parse(&args)).is_err());
            assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        }
        let args = [Value::from("RowGroups"), resident];
        let future = parse(&args);
        drop(future);
        assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let paged = |text: &str| {
                Value::CharArray(
                    CharArray::from_unicode_scalars_with_shape(
                        text.chars().collect(),
                        vec![1, 1, text.chars().count()],
                    )
                    .unwrap(),
                )
            };
            assert!(block_on(parse(&[paged("OutputType"), paged("table"),])).is_ok());
            let parsed = block_on(parse(&args)).unwrap();
            assert_eq!(parsed.row_groups, Some(vec![1]));
            assert_eq!(owner.downloads.load(Ordering::SeqCst), 1);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let error = block_on(parse(&args)).err().unwrap();
            assert_eq!(
                error.identifier(),
                Some("RunMat:readtable:UnsupportedFileType")
            );
            assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
        }
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
    });
}
async fn parse(args: &[Value]) -> BuiltinResult<ParquetReadOptions> {
    ParquetReadOptions::parse(crate::call::FunctionArgumentView::from_slice(args)).await
}

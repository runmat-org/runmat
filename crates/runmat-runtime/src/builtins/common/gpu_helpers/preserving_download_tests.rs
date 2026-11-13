use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_accelerate_api::{GpuTensorLayout, GpuTensorView, TransposeInfo};
use std::sync::Mutex;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn download_diagnostic_owner_preserves_typed_provider_failure_and_cause() {
    use std::error::Error as _;
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    for (diagnostic, expected) in [
        (NumericDownloadDiagnostic::Gpu, "RunMat:gpu:DownloadFailed"),
        (
            NumericDownloadDiagnostic::Gather,
            "RunMat:gather:DownloadFailed",
        ),
    ] {
        let error = finish_native_download(
            &allocation.handle,
            NumericElementType::F64,
            Err(anyhow::anyhow!("injected untyped download failure")),
            diagnostic,
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some(expected));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error
            .source()
            .unwrap()
            .to_string()
            .contains("injected untyped download failure"));
        let original = build_runtime_error("typed provider diagnostic")
            .with_identifier("RunMat:Test:ProviderDownload")
            .with_source(std::io::Error::other("injected typed download cause"))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        let error = finish_native_download(
            &allocation.handle,
            NumericElementType::F64,
            Err(anyhow::Error::new(original)),
            diagnostic,
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Test:ProviderDownload"));
        assert_eq!(error.message(), "typed provider diagnostic");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while let Some(next) = cause.source() {
            cause = next;
        }
        assert_eq!(cause.to_string(), "injected typed download cause");
    }
}

fn release_all(handles: impl IntoIterator<Item = GpuTensorHandle>) {
    let handles: Vec<_> = handles.into_iter().collect();
    let leases: Vec<_> = handles
        .iter()
        .map(|h| h.allocation_lease().downgrade())
        .collect();
    for handle in handles {
        handle.release().expect("release owned fixture");
    }
    for lease in leases {
        assert!(lease.acquire().is_none(), "final physical lease retired");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn native_upload_and_download_preserve_low_rank_shape_class_and_owner() {
    test_support::with_test_provider(|provider| {
        let real = Tensor::new(vec![2.5], Vec::new()).unwrap();
        let real_handle = upload_tensor(provider, &real).expect("rank-zero real upload");
        let real_descriptor = real_handle.wire_descriptor();
        assert!(runmat_accelerate_api::same_provider_instance(
            exact_provider_for_handle(&real_handle).expect("real owner"),
            provider,
        ));
        let native = block_on(download_native_values_async(provider, &real_handle))
            .expect("rank-zero native download");
        assert!(native.shape.is_empty());
        assert_eq!(native.data, vec![NumericScalar::F64(2.5)]);
        assert!(real.shape().is_empty());
        assert_eq!(real_handle.wire_descriptor(), real_descriptor);

        let complex = ComplexTensor::from_f32(vec![(1.25, -2.5), (3.5, 4.25)], vec![2])
            .expect("rank-one complex source");
        let complex_handle =
            upload_complex_tensor(provider, &complex).expect("rank-one complex upload");
        let complex_descriptor = complex_handle.wire_descriptor();
        assert!(runmat_accelerate_api::same_provider_instance(
            exact_provider_for_handle(&complex_handle).expect("complex owner"),
            provider,
        ));
        let downloaded = block_on(download_value_preserving_residency_async(
            provider,
            &complex_handle,
        ))
        .expect("rank-one complex download");
        let Value::ComplexTensor(downloaded) = downloaded else {
            panic!("complex storage retained")
        };
        assert_eq!(downloaded.shape(), &[2]);
        assert_eq!(downloaded.numeric_dtype(), NumericDType::F32);
        assert_eq!(downloaded.complex_storage(), complex.complex_storage());
        assert_eq!(complex.shape(), &[2]);
        assert_eq!(complex_handle.wire_descriptor(), complex_descriptor);
        release_all([real_handle, complex_handle]);
    });
}

struct ReturningUploadProvider<'a> {
    inner: &'a dyn AccelProvider,
    outputs: Mutex<Vec<GpuTensorHandle>>,
}

impl AccelProvider for ReturningUploadProvider<'_> {
    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self
            .outputs
            .lock()
            .unwrap()
            .pop()
            .expect("one result per upload"))
    }
    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        Ok(self
            .outputs
            .lock()
            .unwrap()
            .pop()
            .expect("one result per upload"))
    }
    fn download<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn device_info(&self) -> String {
        "fixed upload result test provider".into()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
}

#[test]
fn protected_restore_retains_explicit_intent_from_any_operand() {
    test_support::with_test_provider(|provider| {
        let host = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let source = upload_tensor(provider, &host)
            .unwrap()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("automatic view");
        let other = upload_tensor(provider, &host)
            .unwrap()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit view");
        let before = other.wire_descriptor();
        let restored = restore_class_preserving_value_protecting(
            &source,
            Value::Tensor(host),
            "test",
            &[&other],
        )
        .unwrap();
        let Value::GpuTensor(output) = restored else {
            panic!("resident output")
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert_eq!(other.wire_descriptor(), before);
        release_all([output, other, source]);
    });
}

#[test]
fn protected_numeric_and_complex_uploads_preserve_borrowed_aliases() {
    test_support::with_test_provider(|owner| {
        let real = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
        for prototype in [
            upload_tensor(owner, &real).unwrap(),
            upload_complex_tensor(owner, &complex).unwrap(),
        ] {
            let before = prototype.wire_descriptor();
            let distinct_view = prototype.with_view(prototype.view().clone()).unwrap();
            assert_ne!(distinct_view.view_identity(), prototype.view_identity());
            assert!(distinct_view.shares_allocation_with(&prototype));
            let provider = ReturningUploadProvider {
                inner: owner,
                outputs: Mutex::new(vec![distinct_view, prototype.clone()]),
            };
            // Both helpers must reject even an otherwise correctly described alias;
            // the opposite storage request also exercises malformed-alias rejection.
            assert!(upload_tensor_protecting(&provider, &real, &[&prototype])
                .unwrap_err()
                .to_string()
                .contains("borrowed input"));
            assert!(
                upload_complex_tensor_protecting(&provider, &complex, &[&prototype])
                    .unwrap_err()
                    .to_string()
                    .contains("borrowed input")
            );
            assert!(provider.outputs.lock().unwrap().is_empty());
            assert_eq!(prototype.wire_descriptor(), before);
            assert!(
                block_on(owner.download_numeric(&prototype)).is_ok(),
                "borrowed allocation remains live"
            );
            release_all([prototype]);
        }
    });
}

#[test]
fn invalid_numeric_upload_releases_actual_allocation_owner_not_invoked_provider() {
    test_support::with_test_provider(|invoked| {
        let alien = TestAllocationOwner::new()
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        assert_ne!(alien.handle.device_id(), invoked.device_id());
        let provider = ReturningUploadProvider {
            inner: invoked,
            outputs: Mutex::new(vec![alien.handle]),
        };
        let host = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        assert!(upload_tensor(&provider, &host).is_err());
        assert!(provider.outputs.lock().unwrap().is_empty());
        assert_eq!(alien.probe.retirements(), 1);
    });
}

#[test]
fn unary_output_contract_checks_owner_class_shape_storage_and_alias_policy() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let input = upload_tensor(provider, &tensor).expect("input upload");
        let output = upload_tensor(provider, &tensor).expect("output upload");
        let contract = UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: Some(ProviderPrecision::F64),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinct,
        };
        assert!(unary_gpu_output_matches(
            &output, &input, provider, contract
        ));
        assert!(!unary_gpu_output_matches(
            &input, &input, provider, contract
        ));
        assert!(unary_gpu_output_matches(
            &input,
            &input,
            provider,
            UnaryGpuOutputContract {
                alias: GpuOutputAliasPolicy::AllowInput,
                ..contract
            }
        ));

        let wrong_shape = output
            .with_view(GpuTensorView::contiguous(vec![2, 1]))
            .unwrap();
        assert!(!unary_gpu_output_matches(
            &wrong_shape,
            &input,
            provider,
            contract
        ));
        let wrong_class = upload_tensor(
            provider,
            &Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap(),
        )
        .unwrap();
        assert!(!unary_gpu_output_matches(
            &wrong_class,
            &input,
            provider,
            contract
        ));
        let wrong_storage = upload_complex_tensor(
            provider,
            &ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0)], vec![1, 2]).unwrap(),
        )
        .unwrap();
        assert!(!unary_gpu_output_matches(
            &wrong_storage,
            &input,
            provider,
            contract
        ));
        let alias_view = input.with_view(input.view().clone()).unwrap();
        assert!(unary_gpu_output_matches(
            &alias_view,
            &input,
            provider,
            contract
        ));
        assert!(!unary_gpu_output_matches(
            &alias_view,
            &input,
            provider,
            UnaryGpuOutputContract {
                alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
                ..contract
            }
        ));
        release_all([
            output,
            input,
            wrong_shape,
            wrong_class,
            wrong_storage,
            alias_view,
        ]);
    });
}

#[test]
fn central_runtime_transfer_round_trips_every_native_real_and_complex_class() {
    test_support::with_test_provider(|provider| {
        let real_cases = vec![
            NumericStorage::F64(vec![-1.25, 2.5]),
            NumericStorage::F32(vec![-1.25, 2.5]),
            NumericStorage::I8(vec![i8::MIN, i8::MAX]),
            NumericStorage::I16(vec![i16::MIN, i16::MAX]),
            NumericStorage::I32(vec![i32::MIN, i32::MAX]),
            NumericStorage::I64(vec![i64::MIN, i64::MAX]),
            NumericStorage::U8(vec![0, u8::MAX]),
            NumericStorage::U16(vec![0, u16::MAX]),
            NumericStorage::U32(vec![0, u32::MAX]),
            NumericStorage::U64(vec![1_u64 << 63, u64::MAX]),
        ];
        for storage in real_cases {
            let expected = storage.clone();
            let tensor = Tensor::from_numeric_storage(storage, vec![1, 2]).unwrap();
            let handle = upload_tensor(provider, &tensor).expect("native real upload");
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(handle.clone()),
            ))
            .expect("native real gather");
            let Value::Tensor(gathered) = gathered else {
                panic!("real transfer reconstructed the wrong value kind")
            };
            assert_eq!(gathered.into_numeric_storage().unwrap(), expected);
            release_all([handle]);
        }

        macro_rules! integer_complex_case {
            ($variant:ident, $real:expr, $imag:expr) => {
                ComplexStorage::Integer(
                    IntegerComplexStorage::new(
                        IntegerStorage::$variant($real),
                        IntegerStorage::$variant($imag),
                    )
                    .unwrap(),
                )
            };
        }
        let complex_cases = vec![
            ComplexStorage::F64(vec![(-1.25, 3.5), (2.5, -4.75)].into()),
            ComplexStorage::F32(vec![(-1.25, 3.5), (2.5, -4.75)].into()),
            integer_complex_case!(I8, vec![i8::MIN, i8::MAX], vec![1, -1]),
            integer_complex_case!(I16, vec![i16::MIN, i16::MAX], vec![1, -1]),
            integer_complex_case!(I32, vec![i32::MIN, i32::MAX], vec![1, -1]),
            integer_complex_case!(I64, vec![i64::MIN, i64::MAX], vec![1, -1]),
            integer_complex_case!(U8, vec![0, u8::MAX], vec![1, 2]),
            integer_complex_case!(U16, vec![0, u16::MAX], vec![1, 2]),
            integer_complex_case!(U32, vec![0, u32::MAX], vec![1, 2]),
            integer_complex_case!(U64, vec![1_u64 << 63, u64::MAX], vec![1, 2]),
        ];
        for storage in complex_cases {
            let expected = storage.clone();
            let tensor = ComplexTensor::from_complex_storage(storage, vec![1, 2]).unwrap();
            let handle = upload_complex_tensor(provider, &tensor).expect("native complex upload");
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(handle.clone()),
            ))
            .expect("native complex gather");
            let Value::ComplexTensor(gathered) = gathered else {
                panic!("complex transfer reconstructed the wrong value kind")
            };
            assert_eq!(gathered.into_complex_storage(), expected);
            release_all([handle]);
        }
    });
}

#[test]
fn class_preserving_restore_uses_shared_single_and_complex_integer_storage() {
    test_support::with_test_provider(|provider| {
        let source =
            upload_tensor(provider, &Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()).unwrap();
        let replacement = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("automatic view");
        drop(source);
        let source = replacement;

        let single = Tensor::from_f32(vec![1.25, -2.5], vec![1, 2]).unwrap();
        let restored =
            restore_class_preserving_value(&source, Value::Tensor(single.clone()), "test")
                .expect("native single restore");
        let Value::GpuTensor(single_handle) = restored else {
            panic!("shared provider should preserve native single residency")
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&single_handle),
            Some(ProviderPrecision::F32)
        );
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(single_handle.clone()),
        ))
        .unwrap();
        assert_eq!(gathered, Value::Tensor(single));

        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                IntegerStorage::U64(vec![3, 4]),
            )
            .unwrap(),
            vec![1, 2],
        )
        .unwrap();
        let restored =
            restore_class_preserving_value(&source, Value::ComplexTensor(complex.clone()), "test")
                .expect("complex integer restore");
        let Value::GpuTensor(complex_handle) = restored else {
            panic!("shared provider should preserve complex integer residency")
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&complex_handle),
            Some(IntegerElementType::U64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_storage(&complex_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(complex_handle.clone()),
        ))
        .unwrap();
        assert_eq!(gathered, Value::ComplexTensor(complex));

        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let restored =
            restore_class_preserving_value(&source, Value::LogicalArray(logical.clone()), "test")
                .expect("logical restore");
        let Value::GpuTensor(logical_handle) = restored else {
            panic!("logical restoration should retain residency")
        };
        assert!(runmat_accelerate_api::handle_is_logical(&logical_handle));
        assert_eq!(
            runmat_accelerate_api::handle_class_identity(&logical_handle),
            Some(runmat_types::standard::LOGICAL.owned())
        );
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(logical_handle.clone()),
        ))
        .unwrap();
        assert_eq!(gathered, Value::LogicalArray(logical));

        release_all([single_handle, complex_handle, logical_handle, source]);
    });
}

#[test]
fn class_preserving_logical_restore_uses_native_logical_encoding() {
    test_support::with_native_logical_test_provider(|provider| {
        let source = upload_tensor(
            provider,
            &Tensor::from_f32(vec![0.0, 0.0], vec![1, 2]).unwrap(),
        )
        .unwrap();
        let replacement = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("automatic view");
        drop(source);
        let source = replacement;
        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let restored =
            restore_class_preserving_value(&source, Value::LogicalArray(logical.clone()), "test")
                .expect("logical restore");
        let Value::GpuTensor(handle) = restored else {
            panic!("logical restoration should retain residency")
        };
        assert_eq!(handle.physical().element_type(), NumericElementType::U8);
        assert!(runmat_accelerate_api::handle_is_logical(&handle));
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(handle.clone()),
        ))
        .unwrap();
        assert_eq!(gathered, Value::LogicalArray(logical));
        release_all([handle, source]);
    });
}

#[test]
fn resident_output_source_prefers_explicit_intent_independent_of_order() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let automatic = upload_tensor(provider, &tensor).unwrap();
        let explicit = upload_tensor(provider, &tensor).unwrap();
        let replacement = automatic
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("automatic view");
        drop(automatic);
        let automatic = replacement;
        let replacement = explicit
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit view");
        drop(explicit);
        let explicit = replacement;
        for handles in [
            vec![automatic.clone(), explicit.clone()],
            vec![explicit.clone(), automatic.clone()],
        ] {
            let selected = select_resident_output_source(handles, "test")
                .unwrap()
                .expect("resident source");
            assert!(same_gpu_handle(&selected, &explicit));
            assert!(runmat_accelerate_api::handle_is_explicit(&selected));
        }
        release_all([automatic, explicit]);
    });
}

#[test]
fn resident_output_source_rejects_a_stale_or_wrong_owner() {
    test_support::with_test_provider(|_| {
        let unregistered = TestAllocationOwner::new()
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let stale = unregistered.handle;
        let error =
            select_resident_output_source([stale], "test").expect_err("unowned handle must reject");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderOwnershipMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        drop(error);
        assert_eq!(unregistered.probe.retirements(), 1);
    });
}

struct FailingNumericUploadProvider {
    owner: TestAllocationOwner,
}

impl AccelProvider for FailingNumericUploadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("only native numeric upload is expected")
    }
    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        Err(anyhow::anyhow!("device unavailable").into())
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        unreachable!("upload failure must not download")
    }
    fn device_info(&self) -> String {
        "failing native upload".into()
    }
}

#[test]
fn exact_integer_scalar_upload_preserves_operational_provider_failures() {
    let provider = FailingNumericUploadProvider {
        owner: TestAllocationOwner::new(),
    };
    let prototype = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::I64, GpuTensorStorage::Real)
        .unwrap();
    let error = upload_exact_integer_scalar_like(&provider, &prototype.handle, 7.0)
        .expect_err("provider failure must not become an inapplicable fast path");
    assert!(error.to_string().contains("device unavailable"));
    assert_eq!(prototype.probe.retirements(), 0);
    drop((error, prototype.handle));
    assert_eq!(prototype.probe.retirements(), 1);
}

#[test]
fn shared_upload_boundary_rejects_wrong_native_class_and_retires_output() {
    test_support::with_test_provider(|owner| {
        let wrong =
            upload_tensor(owner, &Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()).unwrap();
        let lease = wrong.allocation_lease().downgrade();
        let provider = ReturningUploadProvider {
            inner: owner,
            outputs: Mutex::new(vec![wrong]),
        };
        let tensor = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let error = upload_tensor(&provider, &tensor).expect_err("wrong native class");
        let message = error.to_string();
        assert!(message.contains("provider returned an invalid numeric upload descriptor"));
        assert!(message.contains("element F32, storage Real"));
        assert!(message.contains("element Some(F64), storage Some(Real)"));
        drop(error);
        assert!(lease.acquire().is_none());
    });
}

#[derive(Clone, Copy)]
enum DownloadBehavior {
    WrongShape,
    WrongClass,
    ClearResidency,
    Pending,
}

struct ControlledDownloadProvider {
    owner: TestAllocationOwner,
    behavior: DownloadBehavior,
}
impl ControlledDownloadProvider {
    fn new(behavior: DownloadBehavior) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            behavior,
        }
    }
    fn input(&self) -> runmat_accelerate_api::test_support::TestAllocation {
        self.owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0, 2.0]),
                shape: vec![1, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
    }
}
impl AccelProvider for ControlledDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "controlled native download".into()
    }
    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("fixture allocates through actual owner")
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        unreachable!("native download must retain dtype")
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            runmat_accelerate_api::clear_residency(handle).expect("clear only view residency");
            if matches!(self.behavior, DownloadBehavior::Pending) {
                futures::future::pending::<()>().await;
            }
            Ok(HostNumericTensorOwned {
                data: if matches!(self.behavior, DownloadBehavior::WrongClass) {
                    HostNumericDataOwned::F32(vec![1.0, 2.0])
                } else {
                    HostNumericDataOwned::F64(vec![1.0, 2.0])
                },
                shape: if matches!(self.behavior, DownloadBehavior::WrongShape) {
                    vec![2, 1]
                } else {
                    handle.shape().to_vec()
                },
                storage: GpuTensorStorage::Real,
            })
        })
    }
}

#[test]
fn owner_download_preserves_integer_source_residency_and_metadata() {
    test_support::with_test_provider(|provider| {
        let source = Tensor::new_integer(IntegerStorage::U64(vec![3, 4]), vec![1, 2])
            .expect("integer source");
        let handle = upload_tensor(provider, &source).expect("upload integer source");
        runmat_accelerate::ensure_residency_hooks();
        runmat_accelerate_api::mark_residency(&handle);
        let original = handle.wire_descriptor();
        // This owner has no lazy-layout execution route. The former metadata
        // mutation fabricated one; immutable acquisition must reject it.
        assert!(handle
            .with_view(GpuTensorView::contiguous(vec![1, 2]).with_layout(
                GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 1
                })
            ))
            .is_err());
        runmat_accelerate_api::mark_residency(&handle);
        {
            let guard = ResidencyRestoreGuard::new(&handle);
            runmat_accelerate_api::clear_residency(&handle).unwrap();
            guard.restore();
        }
        assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
        assert_eq!(handle.wire_descriptor(), original);
        let metadata = (
            runmat_accelerate_api::handle_storage(&handle),
            runmat_accelerate_api::handle_precision(&handle),
            runmat_accelerate_api::handle_integer_type(&handle),
            runmat_accelerate_api::handle_is_logical(&handle),
        );
        let downloaded = block_on(download_value_preserving_residency_async(provider, &handle))
            .expect("non-destructive owner download");
        let Value::Tensor(downloaded) = downloaded else {
            panic!("expected integer tensor")
        };
        assert_eq!(
            downloaded.into_numeric_storage().unwrap(),
            runmat_value::NumericStorage::U64(vec![3, 4])
        );
        assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
        assert_eq!(runmat_accelerate_api::handle_storage(&handle), metadata.0);
        assert_eq!(runmat_accelerate_api::handle_precision(&handle), metadata.1);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&handle),
            metadata.2
        );
        assert_eq!(
            runmat_accelerate_api::handle_is_logical(&handle),
            metadata.3
        );
        assert_eq!(handle.wire_descriptor(), original);
        release_all([handle]);
    });
}

#[test]
fn native_truth_download_keeps_wide_unsigned_values_exact_and_resident() {
    test_support::with_test_provider(|provider| {
        let source = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 1, 1_u64 << 63, u64::MAX]),
            vec![1, 4],
        )
        .expect("wide integer source");
        let handle = upload_tensor(provider, &source).expect("wide integer upload");
        let replacement = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("automatic view");
        drop(handle);
        let handle = replacement;

        let native = block_on(download_native_values_async(provider, &handle))
            .expect("native-value download");
        assert_eq!(
            native.data,
            vec![
                NumericScalar::U64(0),
                NumericScalar::U64(1),
                NumericScalar::U64(1_u64 << 63),
                NumericScalar::U64(u64::MAX),
            ]
        );
        let truth =
            block_on(download_truth_values_async(provider, &handle)).expect("truth-value download");
        assert_eq!(truth.data, vec![0, 1, 1, 1]);
        assert_eq!(truth.shape, vec![1, 4]);
        assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&handle),
            Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        );

        release_all([handle]);
    });
}

#[test]
fn preserving_download_contract_errors_are_terminal_to_dispatcher_retry() {
    let error =
        build_runtime_error("gpu download: complex-interleaved buffer has odd scalar length")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
}

#[test]
fn preserving_download_restores_residency_and_preserves_immutable_class() {
    let _lock = test_support::accel_test_lock();
    runmat_accelerate::ensure_residency_hooks();
    let provider = ControlledDownloadProvider::new(DownloadBehavior::ClearResidency);
    let allocation = provider.input();
    let original = allocation.handle.wire_descriptor();
    runmat_accelerate_api::mark_residency(&allocation.handle);
    let result = block_on(download_value_preserving_residency_async(
        &provider,
        &allocation.handle,
    ))
    .expect("valid native payload");
    let Value::Tensor(tensor) = result else {
        panic!("double tensor")
    };
    assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
    assert_eq!(tensor.shape(), &[1, 2]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0]);
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert!(!runmat_accelerate_api::handle_is_logical(
        &allocation.handle
    ));
    assert_eq!(
        runmat_accelerate_api::handle_class_identity(&allocation.handle),
        Some(runmat_types::standard::DOUBLE.owned())
    );
    assert!(runmat_accelerate::fusion_residency::is_resident(
        &allocation.handle
    ));
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn preserving_download_rejects_provider_payload_shape_and_class_mismatch() {
    let _lock = test_support::accel_test_lock();
    runmat_accelerate::ensure_residency_hooks();
    for behavior in [DownloadBehavior::WrongShape, DownloadBehavior::WrongClass] {
        let provider = ControlledDownloadProvider::new(behavior);
        let allocation = provider.input();
        let original = allocation.handle.wire_descriptor();
        runmat_accelerate_api::mark_residency(&allocation.handle);
        let error = block_on(download_value_preserving_residency_async(
            &provider,
            &allocation.handle,
        ))
        .expect_err("native payload must match immutable source");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert_eq!(allocation.handle.wire_descriptor(), original);
        assert!(runmat_accelerate::fusion_residency::is_resident(
            &allocation.handle
        ));
        assert_eq!(allocation.probe.retirements(), 0);
        drop((error, allocation.handle));
        assert_eq!(allocation.probe.retirements(), 1);
    }
}

#[test]
fn dispatcher_gather_marks_actual_malformed_provider_downloads() {
    let _lock = test_support::accel_test_lock();
    runmat_accelerate::ensure_residency_hooks();
    for behavior in [DownloadBehavior::WrongShape, DownloadBehavior::WrongClass] {
        // Provider selection requires a process-lifetime reference, just as a
        // registered execution provider does. Each fixture owns a distinct device.
        let provider: &'static ControlledDownloadProvider =
            Box::leak(Box::new(ControlledDownloadProvider::new(behavior)));
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let allocation = provider.input();
        runmat_accelerate_api::mark_residency(&allocation.handle);
        let source = Value::GpuTensor(allocation.handle.clone());
        let error = block_on(crate::dispatcher::gather_if_needed_async(&source))
            .expect_err("dispatcher must reject a malformed provider download");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(error.source.as_deref().is_some_and(|source| {
            source.is::<crate::dispatcher::ProviderDownloadPayloadMismatch>()
        }));
        drop((error, source, allocation.handle));
        assert_eq!(allocation.probe.retirements(), 1);
    }
}

#[test]
fn pending_download_drop_preserves_borrowed_allocation_and_residency() {
    use std::future::Future;
    use std::task::{Context, Poll};
    let _lock = test_support::accel_test_lock();
    runmat_accelerate::ensure_residency_hooks();
    let provider = ControlledDownloadProvider::new(DownloadBehavior::Pending);
    let allocation = provider.input();
    let alias = allocation.handle.clone();
    let original = alias.wire_descriptor();
    runmat_accelerate_api::mark_residency(&alias);
    let mut future = Box::pin(download_value_preserving_residency_async(
        &provider,
        &allocation.handle,
    ));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(allocation.probe.retirements(), 0);
    drop(future);
    assert_eq!(alias.wire_descriptor(), original);
    assert!(runmat_accelerate::fusion_residency::is_resident(&alias));
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(alias);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn preserving_download_requires_owner_admitted_physical_metadata() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let handle = upload_tensor(provider, &tensor).expect("upload");
        let original = handle.wire_descriptor();
        let downloaded = block_on(download_value_preserving_residency_async(provider, &handle))
            .expect("durable descriptor must be accepted");
        assert!(matches!(downloaded, Value::Tensor(_)));
        let mut contradictory = original.clone();
        contradictory.physical = runmat_accelerate_api::GpuPhysicalDescriptor::new(
            NumericElementType::I64,
            GpuTensorStorage::Real,
            1,
            8,
            8,
        )
        .unwrap();
        provider.reset_telemetry();
        assert_eq!(
            provider.acquire_live(&contradictory).unwrap_err(),
            runmat_accelerate_api::allocation::AllocationAcquisitionError::PhysicalMismatch
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(handle.wire_descriptor(), original);
        let again = block_on(download_value_preserving_residency_async(provider, &handle)).unwrap();
        assert_eq!(again, Value::Tensor(tensor));
        release_all([handle]);
    });
}

use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorOwned,
    HostTensorView,
};
use runmat_types::{
    DistributedFact, DistributedValueId, ParallelRegionId, ProgramFunctionId, RegionId, ShapeFact,
};

#[test]
fn native_real64_can_be_borrowed_or_moved_without_copying() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let value = Value::Tensor(tensor.clone());
    let borrowed =
        admit_borrowed_real64(&value, AnalysisValueAdmissionPolicy::exact_host()).unwrap();
    assert_eq!(borrowed.values(), &[1.0, 2.0, 3.0]);
    assert_eq!(borrowed.shape(), &[3, 1]);
    assert_eq!(
        borrowed.receipt().ownership,
        ValueAdmissionOwnership::Borrowed
    );
    assert_eq!(borrowed.receipt().copies, ValueCopyAccounting::default());

    let materialized_borrow =
        admit_borrowed_real64(&value, AnalysisValueAdmissionPolicy::exact_host())
            .unwrap()
            .into_owned();
    assert_eq!(
        materialized_borrow.receipt().ownership,
        ValueAdmissionOwnership::Materialized
    );
    assert_eq!(
        materialized_borrow
            .receipt()
            .copies
            .host_materialization_bytes,
        24
    );

    let moved = admit_owned_real64(
        Value::Tensor(tensor),
        AnalysisValueAdmissionPolicy::exact_host(),
    )
    .unwrap();
    assert_eq!(moved.values(), &[1.0, 2.0, 3.0]);
    assert_eq!(moved.receipt().ownership, ValueAdmissionOwnership::Moved);
    assert_eq!(moved.receipt().copies, ValueCopyAccounting::default());
}

#[test]
fn portable_real64_uses_the_same_owned_admission_and_receipt() {
    let payload = ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
        element_type: ElementType::F64,
        shape: vec![2, 1],
        little_endian_data: [3.0_f64, 5.0]
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
    })));
    let admitted =
        admit_portable_real64(payload.clone(), AnalysisValueAdmissionPolicy::exact_host()).unwrap();
    assert_eq!(admitted.values(), &[3.0, 5.0]);
    assert_eq!(admitted.shape(), &[2, 1]);
    assert_eq!(admitted.receipt().ownership, ValueAdmissionOwnership::Moved);
    admitted.receipt().validate().unwrap();
    let (closed, receipt) = admitted.into_value_payload().unwrap();
    assert_eq!(closed, payload);
    receipt.validate().unwrap();

    let mut wrong_schema = receipt;
    wrong_schema.schema_version = 0;
    assert!(wrong_schema.validate().is_err());

    let scalar = ValuePayload::Inline(Box::new(InlineValue::F64Bits(7.0_f64.to_bits())));
    let admitted =
        admit_portable_real64(scalar.clone(), AnalysisValueAdmissionPolicy::exact_host()).unwrap();
    assert_eq!(admitted.receipt().target.storage, StorageFact::Scalar);
    assert_eq!(admitted.into_value_payload().unwrap().0, scalar);
}

#[test]
fn conversion_is_forbidden_unless_explicit_and_accounts_for_materialization() {
    let value = Value::Tensor(Tensor::from_f32(vec![1.25, 2.5], vec![1, 2]).unwrap());
    let error =
        admit_borrowed_real64(&value, AnalysisValueAdmissionPolicy::exact_host()).unwrap_err();
    assert_eq!(
        error,
        AnalysisValueAdmissionError::ExplicitNumericConversionRequired {
            source_class: NumericClass::Single
        }
    );

    let admitted = admit_borrowed_real64(
        &value,
        AnalysisValueAdmissionPolicy::explicit_host_conversion(),
    )
    .unwrap();
    assert_eq!(admitted.values(), &[1.25, 2.5]);
    assert_eq!(admitted.receipt().copies.host_materialization_bytes, 16);
    assert_eq!(
        admitted.receipt().conversion,
        Some(ValueNumericConversion {
            source: NumericClass::Single,
            target: NumericClass::Double,
            fidelity: ValueConversionFidelity::Exact,
        })
    );
}

#[test]
fn sparse_complex_and_distributed_facts_fail_closed() {
    let sparse = runmat_value::SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
    assert_eq!(
        admit_borrowed_real64(
            &Value::SparseTensor(sparse),
            AnalysisValueAdmissionPolicy::explicit_host_conversion(),
        )
        .unwrap_err(),
        AnalysisValueAdmissionError::SparseDensificationRequired
    );
    assert_eq!(
        admit_borrowed_real64(
            &Value::Complex(1.0, 2.0),
            AnalysisValueAdmissionPolicy::explicit_host_conversion(),
        )
        .unwrap_err(),
        AnalysisValueAdmissionError::ComplexDomain
    );

    let mut distributed = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    distributed.kind = ValueKindFact::Distributed(DistributedFact {
        id: DistributedValueId {
            function: ProgramFunctionId(0),
            ordinal: 1,
        },
        owner: runmat_types::DistributedOwner::Region(ParallelRegionId(RegionId {
            function: ProgramFunctionId(0),
            ordinal: 2,
        })),
        scheme: None,
        value: Box::new(value_fact(&Value::Num(0.0))),
        materializable: true,
    });
    distributed.shape = ShapeFact::Unknown;
    assert_eq!(
        preflight_real64_admission(
            &distributed,
            AnalysisValueAdmissionPolicy::explicit_host_conversion_and_download(),
        )
        .unwrap_err(),
        AnalysisValueAdmissionError::DistributedRedistributionRequired
    );

    let mut strided = value_fact(&Value::Tensor(
        Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
    ));
    strided.layout = LayoutFact::Strided;
    strided.contiguity = ContiguityFact::NonContiguous;
    assert_eq!(
        preflight_real64_admission(
            &strided,
            AnalysisValueAdmissionPolicy::explicit_host_conversion(),
        )
        .unwrap_err(),
        AnalysisValueAdmissionError::HostLayoutMaterializationRequired {
            layout: LayoutFact::Strided,
            contiguity: ContiguityFact::NonContiguous,
        }
    );
}

/// Owns real fixture allocations; downloads return the admitted values.
struct DownloadProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
}

impl AccelProvider for DownloadProvider {
    fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("the admission test only downloads")
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            Ok(HostTensorOwned {
                data: vec![4.0, 5.0],
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }

    fn device_info(&self) -> String {
        "analysis-value-admission-test".to_string()
    }

    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
}

#[test]
fn device_download_is_explicit_and_accounts_for_transferred_bytes() {
    let provider: &'static DownloadProvider = Box::leak(Box::new(DownloadProvider {
        owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
    }));
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let handle = provider
        .owner
        .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
            data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![4.0, 5.0]),
            shape: vec![2, 1],
            storage: GpuTensorStorage::Real,
        })
        .expect("fixture allocation")
        .handle;
    let value = Value::GpuTensor(handle.clone());

    let rejected = pollster::block_on(admit_owned_real64_async(
        value.clone(),
        AnalysisValueAdmissionPolicy::exact_host(),
    ))
    .unwrap_err();
    assert_eq!(
        rejected,
        AnalysisValueAdmissionError::DeviceTransferRequired
    );

    let admitted = pollster::block_on(admit_owned_real64_async(
        value,
        AnalysisValueAdmissionPolicy::explicit_host_conversion_and_download(),
    ))
    .unwrap();
    assert_eq!(admitted.values(), &[4.0, 5.0]);
    assert_eq!(
        admitted.receipt().transfer,
        ValueAdmissionTransfer::DeviceToHost {
            device_id: provider.device_id()
        }
    );
    assert_eq!(admitted.receipt().copies.device_to_host_bytes, 16);
    assert_eq!(admitted.receipt().copies.host_materialization_bytes, 0);
    assert_eq!(
        admitted.receipt().ownership,
        ValueAdmissionOwnership::Materialized
    );

    assert_eq!(handle.device_id(), provider.device_id());
}

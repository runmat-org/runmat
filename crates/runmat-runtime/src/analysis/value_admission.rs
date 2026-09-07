//! Typed admission of live RunMat numeric values into analysis-owned real64 arrays.
//!
//! This is a runtime boundary: numerical kernels consume the returned slices or vectors and never
//! depend on [`runmat_value::Value`]. Every representation change is explicit in the policy and
//! receipt. In particular, device downloads and integer/single conversion cannot happen through
//! the exact-host policy.

use std::borrow::Cow;

use runmat_types::{
    AliasFact, CertaintyFact, ContiguityFact, LayoutFact, MutationFact, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
};
use runmat_value::{IntValue, NumericDType, NumericStorage, Tensor, Value};
use serde::{Deserialize, Serialize};

use crate::builtins::common::gpu_helpers::{
    download_value_preserving_residency_async, exact_provider_for_handle,
};
use crate::value_fact::value_fact;

const SCALAR_SHAPE: [usize; 2] = [1, 1];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Real64ConversionPolicy {
    #[default]
    Exact,
    Explicit,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceTransferPolicy {
    #[default]
    Reject,
    DownloadToHost,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisValueAdmissionPolicy {
    pub real64_conversion: Real64ConversionPolicy,
    pub device_transfer: DeviceTransferPolicy,
}

impl AnalysisValueAdmissionPolicy {
    pub const fn exact_host() -> Self {
        Self {
            real64_conversion: Real64ConversionPolicy::Exact,
            device_transfer: DeviceTransferPolicy::Reject,
        }
    }

    pub const fn explicit_host_conversion() -> Self {
        Self {
            real64_conversion: Real64ConversionPolicy::Explicit,
            device_transfer: DeviceTransferPolicy::Reject,
        }
    }

    pub const fn explicit_host_conversion_and_download() -> Self {
        Self {
            real64_conversion: Real64ConversionPolicy::Explicit,
            device_transfer: DeviceTransferPolicy::DownloadToHost,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisValueOwnership {
    Borrowed,
    Moved,
    Materialized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericConversionFidelity {
    Exact,
    PotentiallyRounded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericConversionReceipt {
    pub source: NumericClass,
    pub target: NumericClass,
    pub fidelity: NumericConversionFidelity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisValueTransfer {
    None,
    DeviceToHost { device_id: u32 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisValueCopyAccounting {
    /// Bytes allocated and written to change the host numeric representation or ownership.
    pub host_materialization_bytes: u64,
    /// Physical numeric bytes copied from a provider-owned device buffer to host storage.
    pub device_to_host_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisValueAdmissionReceipt {
    pub source: ValueFact,
    pub target: ValueFact,
    pub ownership: AnalysisValueOwnership,
    pub conversion: Option<NumericConversionReceipt>,
    pub transfer: AnalysisValueTransfer,
    pub copies: AnalysisValueCopyAccounting,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum AnalysisValueAdmissionError {
    #[error("analysis admission requires a real numeric value, got {kind:?}")]
    UnsupportedValueKind { kind: ValueKindFact },
    #[error("analysis real64 admission does not accept complex numeric values")]
    ComplexDomain,
    #[error("analysis dense admission does not implicitly densify sparse values")]
    SparseDensificationRequired,
    #[error("analysis host admission does not support {storage:?} storage")]
    UnsupportedHostStorage { storage: StorageFact },
    #[error("analysis real64 admission requires an explicit conversion from {source_class:?}")]
    ExplicitNumericConversionRequired { source_class: NumericClass },
    #[error("analysis host admission requires an explicit device-to-host transfer")]
    DeviceTransferRequired,
    #[error("device download requires the asynchronous owned admission entrypoint")]
    AsyncDeviceAdmissionRequired,
    #[error("no acceleration provider owns device {device_id}")]
    DeviceOwnerUnavailable { device_id: u32 },
    #[error("device {device_id} download failed: {detail}")]
    DeviceDownloadFailed { device_id: u32, detail: String },
    #[error("distributed analysis admission requires an explicit redistribution plan")]
    DistributedRedistributionRequired,
    #[error("remote analysis admission requires an explicit artifact or redistribution plan")]
    RemoteMaterializationRequired,
    #[error("analysis admission cannot prove the source residency")]
    UnprovenResidency,
    #[error(
        "analysis host admission requires explicit layout materialization from {layout:?}/{contiguity:?}"
    )]
    HostLayoutMaterializationRequired {
        layout: LayoutFact,
        contiguity: ContiguityFact,
    },
    #[error("numeric array shape {shape:?} overflows the platform element count")]
    ShapeOverflow { shape: Vec<usize> },
    #[error("numeric byte length overflows for {element_count} elements of {element_size} bytes")]
    ByteLengthOverflow {
        element_count: usize,
        element_size: usize,
    },
    #[error("numeric storage is inconsistent with its shape: {detail}")]
    InvalidStorage { detail: String },
}

#[derive(Debug)]
pub struct AdmittedReal64Array<'a> {
    values: Cow<'a, [f64]>,
    shape: Cow<'a, [usize]>,
    receipt: AnalysisValueAdmissionReceipt,
}

impl<'a> AdmittedReal64Array<'a> {
    pub fn values(&self) -> &[f64] {
        self.values.as_ref()
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_ref()
    }

    pub fn receipt(&self) -> &AnalysisValueAdmissionReceipt {
        &self.receipt
    }

    pub fn into_owned(mut self) -> OwnedAdmittedReal64Array {
        let was_borrowed = matches!(self.values, Cow::Borrowed(_));
        let values = self.values.into_owned();
        if was_borrowed {
            self.receipt.ownership = AnalysisValueOwnership::Materialized;
            self.receipt.copies.host_materialization_bytes = byte_len(values.len(), 8)
                .expect("an admitted array already has a validated element count");
            self.receipt.target.view = ViewFact::Materialized;
            self.receipt.target.alias = AliasFact::Unique;
        }
        OwnedAdmittedReal64Array {
            values,
            shape: self.shape.into_owned(),
            receipt: self.receipt,
        }
    }
}

#[derive(Debug)]
pub struct OwnedAdmittedReal64Array {
    values: Vec<f64>,
    shape: Vec<usize>,
    receipt: AnalysisValueAdmissionReceipt,
}

impl OwnedAdmittedReal64Array {
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn receipt(&self) -> &AnalysisValueAdmissionReceipt {
        &self.receipt
    }

    pub fn into_parts(self) -> (Vec<f64>, Vec<usize>, AnalysisValueAdmissionReceipt) {
        (self.values, self.shape, self.receipt)
    }
}

/// Check whether a static runtime fact can enter the dense host-real64 boundary under `policy`.
///
/// Planners can call this before a live value exists. A successful preflight does not replace the
/// live storage and descriptor validation performed by the admission entrypoints.
pub fn preflight_real64_admission(
    source: &ValueFact,
    policy: AnalysisValueAdmissionPolicy,
) -> Result<(), AnalysisValueAdmissionError> {
    if matches!(source.kind, ValueKindFact::Distributed(_)) {
        return Err(AnalysisValueAdmissionError::DistributedRedistributionRequired);
    }
    match source.residency {
        ResidencyFact::Remote { .. } => {
            return Err(AnalysisValueAdmissionError::RemoteMaterializationRequired)
        }
        ResidencyFact::Device { .. } if policy.device_transfer == DeviceTransferPolicy::Reject => {
            return Err(AnalysisValueAdmissionError::DeviceTransferRequired)
        }
        ResidencyFact::Unknown => return Err(AnalysisValueAdmissionError::UnprovenResidency),
        ResidencyFact::Host | ResidencyFact::Device { .. } => {}
    }
    if source.residency == ResidencyFact::Host
        && (source.layout != LayoutFact::ColumnMajor
            || source.contiguity != ContiguityFact::Contiguous)
    {
        return Err(
            AnalysisValueAdmissionError::HostLayoutMaterializationRequired {
                layout: source.layout,
                contiguity: source.contiguity,
            },
        );
    }
    if source.storage == StorageFact::Sparse {
        return Err(AnalysisValueAdmissionError::SparseDensificationRequired);
    }
    if source.residency == ResidencyFact::Host
        && !matches!(source.storage, StorageFact::Scalar | StorageFact::Dense)
    {
        return Err(AnalysisValueAdmissionError::UnsupportedHostStorage {
            storage: source.storage,
        });
    }
    let Some(numeric) = source.numeric() else {
        return Err(AnalysisValueAdmissionError::UnsupportedValueKind {
            kind: source.kind.clone(),
        });
    };
    if numeric.domain != NumericDomain::Real {
        return Err(AnalysisValueAdmissionError::ComplexDomain);
    }
    if numeric.class != NumericClass::Double
        && policy.real64_conversion == Real64ConversionPolicy::Exact
    {
        return Err(
            AnalysisValueAdmissionError::ExplicitNumericConversionRequired {
                source_class: numeric.class,
            },
        );
    }
    Ok(())
}

/// Admit a live host value, borrowing native real64 storage whenever possible.
pub fn admit_borrowed_real64<'a>(
    value: &'a Value,
    policy: AnalysisValueAdmissionPolicy,
) -> Result<AdmittedReal64Array<'a>, AnalysisValueAdmissionError> {
    let source = value_fact(value);
    preflight_real64_admission(&source, policy)?;
    match value {
        Value::Num(value) => Ok(borrowed(std::slice::from_ref(value), &SCALAR_SHAPE, source)),
        Value::Int(value) => Ok(materialized(
            vec![value.to_f64()],
            SCALAR_SHAPE.to_vec(),
            source,
            Some(integer_conversion(value)),
        )?),
        Value::Tensor(tensor) => admit_borrowed_tensor(tensor, source),
        Value::GpuTensor(_) => Err(AnalysisValueAdmissionError::AsyncDeviceAdmissionRequired),
        _ => Err(AnalysisValueAdmissionError::UnsupportedValueKind { kind: source.kind }),
    }
}

/// Admit a consumed host value, moving native real64 storage whenever possible.
pub fn admit_owned_real64(
    value: Value,
    policy: AnalysisValueAdmissionPolicy,
) -> Result<OwnedAdmittedReal64Array, AnalysisValueAdmissionError> {
    let source = value_fact(&value);
    preflight_real64_admission(&source, policy)?;
    match value {
        Value::Num(value) => materialized_owned(vec![value], SCALAR_SHAPE.to_vec(), source, None),
        Value::Int(value) => materialized_owned(
            vec![value.to_f64()],
            SCALAR_SHAPE.to_vec(),
            source,
            Some(integer_conversion(&value)),
        ),
        Value::Tensor(tensor) => admit_owned_tensor(tensor, source),
        Value::GpuTensor(_) => Err(AnalysisValueAdmissionError::AsyncDeviceAdmissionRequired),
        _ => Err(AnalysisValueAdmissionError::UnsupportedValueKind { kind: source.kind }),
    }
}

/// Admit a consumed value and perform a policy-authorized provider download when it is resident.
pub async fn admit_owned_real64_async(
    value: Value,
    policy: AnalysisValueAdmissionPolicy,
) -> Result<OwnedAdmittedReal64Array, AnalysisValueAdmissionError> {
    let Value::GpuTensor(handle) = value else {
        return admit_owned_real64(value, policy);
    };
    let source = value_fact(&Value::GpuTensor(handle.clone()));
    preflight_real64_admission(&source, policy)?;
    let provider = exact_provider_for_handle(&handle).ok_or(
        AnalysisValueAdmissionError::DeviceOwnerUnavailable {
            device_id: handle.device_id,
        },
    )?;
    let element_type = handle.descriptor.element_type.ok_or_else(|| {
        AnalysisValueAdmissionError::DeviceDownloadFailed {
            device_id: handle.device_id,
            detail: "numeric handle is missing its durable element descriptor".to_string(),
        }
    })?;
    let element_count = checked_element_count(&handle.shape)?;
    let transfer_bytes = byte_len(element_count, element_type.element_size())?;
    let downloaded = download_value_preserving_residency_async(provider, &handle)
        .await
        .map_err(|error| AnalysisValueAdmissionError::DeviceDownloadFailed {
            device_id: handle.device_id,
            detail: error.to_string(),
        })?;
    let mut admitted = admit_owned_real64(
        downloaded,
        AnalysisValueAdmissionPolicy {
            real64_conversion: policy.real64_conversion,
            device_transfer: DeviceTransferPolicy::Reject,
        },
    )?;
    admitted.receipt.source = source;
    admitted.receipt.ownership = AnalysisValueOwnership::Materialized;
    admitted.receipt.transfer = AnalysisValueTransfer::DeviceToHost {
        device_id: handle.device_id,
    };
    admitted.receipt.copies.device_to_host_bytes = transfer_bytes;
    Ok(admitted)
}

fn admit_borrowed_tensor<'a>(
    tensor: &'a Tensor,
    source: ValueFact,
) -> Result<AdmittedReal64Array<'a>, AnalysisValueAdmissionError> {
    if let Some(values) = tensor.as_f64_slice() {
        return Ok(borrowed(values, &tensor.shape, source));
    }
    materialized(
        tensor.materialize_f64(),
        tensor.shape.clone(),
        source,
        Some(dtype_conversion(tensor.numeric_dtype())),
    )
}

fn admit_owned_tensor(
    tensor: Tensor,
    source: ValueFact,
) -> Result<OwnedAdmittedReal64Array, AnalysisValueAdmissionError> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|detail| AnalysisValueAdmissionError::InvalidStorage { detail })?;
    match storage {
        NumericStorage::F64(values) => Ok(OwnedAdmittedReal64Array {
            values,
            shape,
            receipt: receipt(source, AnalysisValueOwnership::Moved, None, 0),
        }),
        storage => {
            let conversion = dtype_conversion(storage.numeric_dtype());
            let values = storage.materialize_f64();
            materialized_owned(values, shape, source, Some(conversion))
        }
    }
}

fn borrowed<'a>(
    values: &'a [f64],
    shape: &'a [usize],
    source: ValueFact,
) -> AdmittedReal64Array<'a> {
    AdmittedReal64Array {
        values: Cow::Borrowed(values),
        shape: Cow::Borrowed(shape),
        receipt: receipt(source, AnalysisValueOwnership::Borrowed, None, 0),
    }
}

fn materialized(
    values: Vec<f64>,
    shape: Vec<usize>,
    source: ValueFact,
    conversion: Option<NumericConversionReceipt>,
) -> Result<AdmittedReal64Array<'static>, AnalysisValueAdmissionError> {
    let bytes = byte_len(values.len(), 8)?;
    Ok(AdmittedReal64Array {
        values: Cow::Owned(values),
        shape: Cow::Owned(shape),
        receipt: receipt(
            source,
            AnalysisValueOwnership::Materialized,
            conversion,
            bytes,
        ),
    })
}

fn materialized_owned(
    values: Vec<f64>,
    shape: Vec<usize>,
    source: ValueFact,
    conversion: Option<NumericConversionReceipt>,
) -> Result<OwnedAdmittedReal64Array, AnalysisValueAdmissionError> {
    let bytes = byte_len(values.len(), 8)?;
    Ok(OwnedAdmittedReal64Array {
        values,
        shape,
        receipt: receipt(
            source,
            AnalysisValueOwnership::Materialized,
            conversion,
            bytes,
        ),
    })
}

fn receipt(
    source: ValueFact,
    ownership: AnalysisValueOwnership,
    conversion: Option<NumericConversionReceipt>,
    host_materialization_bytes: u64,
) -> AnalysisValueAdmissionReceipt {
    let mut target = source.clone();
    target.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    target.storage = if target.shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    target.layout = LayoutFact::ColumnMajor;
    target.contiguity = ContiguityFact::Contiguous;
    target.view = if ownership == AnalysisValueOwnership::Borrowed {
        ViewFact::ReadOnlyView
    } else {
        ViewFact::Materialized
    };
    target.residency = ResidencyFact::Host;
    target.alias = if ownership == AnalysisValueOwnership::Borrowed {
        AliasFact::Shared
    } else {
        AliasFact::Unique
    };
    target.mutation = MutationFact::Immutable;
    target.certainty = CertaintyFact::Proven;
    AnalysisValueAdmissionReceipt {
        source,
        target,
        ownership,
        conversion,
        transfer: AnalysisValueTransfer::None,
        copies: AnalysisValueCopyAccounting {
            host_materialization_bytes,
            device_to_host_bytes: 0,
        },
    }
}

fn dtype_conversion(source: NumericDType) -> NumericConversionReceipt {
    NumericConversionReceipt {
        source: numeric_class(source),
        target: NumericClass::Double,
        fidelity: match source {
            NumericDType::I64 | NumericDType::U64 => NumericConversionFidelity::PotentiallyRounded,
            NumericDType::F64
            | NumericDType::F32
            | NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32 => NumericConversionFidelity::Exact,
        },
    }
}

fn integer_conversion(source: &IntValue) -> NumericConversionReceipt {
    dtype_conversion(match source {
        IntValue::I8(_) => NumericDType::I8,
        IntValue::I16(_) => NumericDType::I16,
        IntValue::I32(_) => NumericDType::I32,
        IntValue::I64(_) => NumericDType::I64,
        IntValue::U8(_) => NumericDType::U8,
        IntValue::U16(_) => NumericDType::U16,
        IntValue::U32(_) => NumericDType::U32,
        IntValue::U64(_) => NumericDType::U64,
    })
}

fn numeric_class(source: NumericDType) -> NumericClass {
    match source {
        NumericDType::F64 => NumericClass::Double,
        NumericDType::F32 => NumericClass::Single,
        NumericDType::I8 => NumericClass::Int8,
        NumericDType::I16 => NumericClass::Int16,
        NumericDType::I32 => NumericClass::Int32,
        NumericDType::I64 => NumericClass::Int64,
        NumericDType::U8 => NumericClass::UInt8,
        NumericDType::U16 => NumericClass::UInt16,
        NumericDType::U32 => NumericClass::UInt32,
        NumericDType::U64 => NumericClass::UInt64,
    }
}

fn checked_element_count(shape: &[usize]) -> Result<usize, AnalysisValueAdmissionError> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| AnalysisValueAdmissionError::ShapeOverflow {
            shape: shape.to_vec(),
        })
}

fn byte_len(element_count: usize, element_size: usize) -> Result<u64, AnalysisValueAdmissionError> {
    let overflow = || AnalysisValueAdmissionError::ByteLengthOverflow {
        element_count,
        element_size,
    };
    let bytes = element_count
        .checked_mul(element_size)
        .ok_or_else(overflow)?;
    u64::try_from(bytes).map_err(|_| overflow())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorOwned,
        HostTensorView, NumericElementType,
    };
    use runmat_types::{
        DistributedFact, DistributedValueId, ParallelRegionId, ProgramFunctionId, RegionId,
        ShapeFact,
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
            AnalysisValueOwnership::Borrowed
        );
        assert_eq!(
            borrowed.receipt().copies,
            AnalysisValueCopyAccounting::default()
        );

        let materialized_borrow =
            admit_borrowed_real64(&value, AnalysisValueAdmissionPolicy::exact_host())
                .unwrap()
                .into_owned();
        assert_eq!(
            materialized_borrow.receipt().ownership,
            AnalysisValueOwnership::Materialized
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
        assert_eq!(moved.receipt().ownership, AnalysisValueOwnership::Moved);
        assert_eq!(
            moved.receipt().copies,
            AnalysisValueCopyAccounting::default()
        );
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
            Some(NumericConversionReceipt {
                source: NumericClass::Single,
                target: NumericClass::Double,
                fidelity: NumericConversionFidelity::Exact,
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
            owner: ParallelRegionId(RegionId {
                function: ProgramFunctionId(0),
                ordinal: 2,
            }),
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

    struct DownloadProvider;

    impl AccelProvider for DownloadProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            unreachable!("the admission test only downloads")
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![4.0, 5.0],
                    shape: handle.shape.clone(),
                    storage: GpuTensorStorage::Real,
                })
            })
        }

        fn free(&self, _handle: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "analysis-value-admission-test".to_string()
        }

        fn device_id(&self) -> u32 {
            9901
        }
    }

    #[test]
    fn device_download_is_explicit_and_accounts_for_transferred_bytes() {
        static PROVIDER: DownloadProvider = DownloadProvider;
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
        let handle = GpuTensorHandle::new(vec![2, 1], PROVIDER.device_id(), 7)
            .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real);
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
            AnalysisValueTransfer::DeviceToHost {
                device_id: PROVIDER.device_id()
            }
        );
        assert_eq!(admitted.receipt().copies.device_to_host_bytes, 16);
        assert_eq!(admitted.receipt().copies.host_materialization_bytes, 0);
        assert_eq!(
            admitted.receipt().ownership,
            AnalysisValueOwnership::Materialized
        );

        assert_eq!(handle.device_id, PROVIDER.device_id());
    }
}

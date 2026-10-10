//! Request-local resident carrier authority. Diagnostic descriptors and global
//! opaque GPU facts alone cannot mint this non-serializable borrowed evidence.

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorLayout, GpuTensorStorage, NumericElementType,
    ProviderElementType, ProviderLayout, ProviderPrecision, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};
use runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle;
use runmat_types::{NumericClass, NumericDomain, StorageFact, ValueFact, ValueKindFact};
use runmat_value::Value;

use crate::graph::ValueId;
use crate::placement::tensor_representation;

pub(super) struct FusionResidentInput<'a> {
    id: ValueId,
    handle: &'a GpuTensorHandle,
    owner: &'static dyn AccelProvider,
    representation: ProviderRepresentation,
    actual_fact: ValueFact,
}

impl<'a> FusionResidentInput<'a> {
    pub(super) fn require_live(&self) -> anyhow::Result<()> {
        self.handle.allocation_lease().validate_live()?;
        self.handle.view().validate(self.handle.physical())?;
        if !exact_provider_for_handle(self.handle)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, self.owner))
        {
            return Err(anyhow::anyhow!("fusion resident input owner changed"));
        }
        Ok(())
    }

    pub(super) fn authenticate(id: ValueId, value: &'a Value) -> Option<Self> {
        let Value::GpuTensor(handle) = value else {
            return None;
        };
        let owner = exact_provider_for_handle(handle)?;
        handle.allocation_lease().validate_live().ok()?;
        handle.view().validate(handle.physical()).ok()?;
        if handle.physical().storage() != GpuTensorStorage::Real
            || handle.view().is_logical()
            || handle.view().layout() != GpuTensorLayout::Contiguous
        {
            return None;
        }
        let (precision, class, element_type) = match handle.physical().element_type() {
            NumericElementType::F32 => (
                ProviderPrecision::F32,
                NumericClass::Single,
                ProviderElementType::F32,
            ),
            NumericElementType::F64 => (
                ProviderPrecision::F64,
                NumericClass::Double,
                ProviderElementType::F64,
            ),
            _ => return None,
        };
        // Reuse the ordinary placement projection; never substitute ambient
        // precision for the immutable physical element class.
        let representation = tensor_representation(value, precision)?;
        let actual_fact = runmat_runtime::value_fact::value_fact(value);
        if representation.element_type != element_type
            || representation.storage != ProviderStorage::DenseReal
            || representation.layout != ProviderLayout::ColumnMajorContiguous
            || representation.residency != ProviderResidency::Device
            || representation.checked_byte_len().is_none()
            || !matches!(&actual_fact.kind, ValueKindFact::Numeric(numeric)
                if numeric.class == class && numeric.domain == NumericDomain::Real)
            || actual_fact.shape.known_dims()?
                != handle.shape().iter().copied().map(Some).collect::<Vec<_>>()
        {
            return None;
        }
        Some(Self {
            id,
            handle,
            owner,
            representation,
            actual_fact,
        })
    }

    pub(super) fn refine_storage(&self, id: ValueId, value: &Value, fact: &mut ValueFact) {
        let Value::GpuTensor(handle) = value else {
            return;
        };
        // Borrowing the real handle retains the owner-admitted view across
        // suspension. A lost owner still invalidates use; equal shape/class or
        // a sibling allocation view cannot substitute for this input binding.
        if id != self.id
            || handle.view_identity() != self.handle.view_identity()
            || fact.kind != self.actual_fact.kind
            || fact.shape != self.actual_fact.shape
            || self.handle.allocation_lease().validate_live().is_err()
            || self.handle.view().validate(self.handle.physical()).is_err()
            || !exact_provider_for_handle(self.handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, self.owner)
            })
            || self.representation.storage != ProviderStorage::DenseReal
        {
            return;
        }
        // No uniqueness, view, layout, sign, payload or initialization proof.
        // The physical/view certificate stays separate from opted-class facts.
        fact.storage = StorageFact::Dense;
    }
}

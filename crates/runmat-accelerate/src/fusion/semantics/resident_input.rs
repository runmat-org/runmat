//! Request-local resident carrier authority. Diagnostic descriptors and global
//! opaque GPU facts alone cannot mint this non-serializable borrowed evidence.

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorLayout, GpuTensorStorage, NumericElementType,
    ProviderElementType, ProviderLayout, ProviderPrecision, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};
use runmat_runtime::builtins::common::gpu_helpers::exact_provider_for_handle;
use runmat_types::{
    DimensionFact, NumericClass, NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use runmat_value::Value;

use crate::graph::ValueId;
use crate::placement::tensor_representation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloatingFusionEligibility {
    Eligible(ProviderPrecision),
    Ineligible,
}

/// Request-local proof that a resident value has a live, internally consistent
/// allocation and one registered owner. Class eligibility is deliberately a
/// separate question: a logical or complex value can be valid resident state
/// without being an input to the real-floating fusion lane.
pub(crate) struct AuthenticatedResidentInput<'a> {
    handle: &'a GpuTensorHandle,
    owner: &'static dyn AccelProvider,
}

impl AuthenticatedResidentInput<'_> {
    pub(crate) fn belongs_to(&self, provider: &dyn AccelProvider) -> bool {
        runmat_accelerate_api::same_provider_instance(self.owner, provider)
    }

    pub(crate) fn floating_eligibility(&self) -> FloatingFusionEligibility {
        if self.handle.view().is_logical()
            || self.handle.physical().storage() != GpuTensorStorage::Real
        {
            return FloatingFusionEligibility::Ineligible;
        }
        match self.handle.physical().element_type() {
            NumericElementType::F32 => FloatingFusionEligibility::Eligible(ProviderPrecision::F32),
            NumericElementType::F64 => FloatingFusionEligibility::Eligible(ProviderPrecision::F64),
            _ => FloatingFusionEligibility::Ineligible,
        }
    }

    pub(crate) fn handle(&self) -> &GpuTensorHandle {
        self.handle
    }
}

pub(crate) fn authenticate_resident_input(
    value: &Value,
) -> anyhow::Result<Option<AuthenticatedResidentInput<'_>>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    handle.allocation_lease().validate_live()?;
    handle.view().validate(handle.physical())?;
    let owner = exact_provider_for_handle(handle)
        .ok_or_else(|| anyhow::anyhow!("fusion resident input has no registered owner"))?;
    Ok(Some(AuthenticatedResidentInput { handle, owner }))
}

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
        let authenticated = authenticate_resident_input(value).ok()??;
        let handle = authenticated.handle;
        let (precision, class, element_type) = match authenticated.floating_eligibility() {
            FloatingFusionEligibility::Eligible(ProviderPrecision::F32) => (
                ProviderPrecision::F32,
                NumericClass::Single,
                ProviderElementType::F32,
            ),
            FloatingFusionEligibility::Eligible(ProviderPrecision::F64) => (
                ProviderPrecision::F64,
                NumericClass::Double,
                ProviderElementType::F64,
            ),
            FloatingFusionEligibility::Ineligible => return None,
        };
        // Reuse the ordinary placement projection; never substitute ambient
        // precision for the immutable physical element class.
        let representation = tensor_representation(value, precision)?;
        let actual_fact = runmat_runtime::value_fact::value_fact(value);
        // The admitted view API has two layouts. A matrix transpose is an
        // authenticated dense carrier with different logical coordinates, not
        // an arbitrary strided descriptor supplied by a caller.
        let layout = match handle.view().layout() {
            GpuTensorLayout::Contiguous => ProviderLayout::ColumnMajorContiguous,
            GpuTensorLayout::Transpose(_) => ProviderLayout::Strided,
        };
        if representation.element_type != element_type
            || representation.storage != ProviderStorage::DenseReal
            || representation.layout != layout
            || representation.residency != ProviderResidency::Device
            || representation.checked_byte_len().is_none()
            || !matches!(&actual_fact.kind, ValueKindFact::Numeric(numeric)
                if numeric.class == class && numeric.domain == NumericDomain::Real)
            || !exact_shape(&actual_fact.shape, handle.shape())
        {
            return None;
        }
        Some(Self {
            id,
            handle,
            owner: authenticated.owner,
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

fn exact_shape(fact: &ShapeFact, dimensions: &[usize]) -> bool {
    match fact {
        ShapeFact::Scalar => dimensions == [1, 1],
        ShapeFact::Shaped { dims } => {
            dims.len() == dimensions.len()
                && dims.iter().zip(dimensions).all(|(fact, actual)| {
                    matches!(fact, DimensionFact::Known(expected) if expected == actual)
                })
        }
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => false,
    }
}

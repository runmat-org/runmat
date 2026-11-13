//! Request-local host geometry for a bounded unpriced physical route.
//! This is distinct from the domain's already-owned source/program payload.

use std::mem::size_of;

use runmat_accelerate_api::{AccelProvider, ProviderFeasibilityQuery, ProviderPreparationRoute};
use runmat_execution::{
    DomainContribution, ExecutionCandidateKind, PlacementOperationId, ProviderResourceSnapshot,
    SelectedExecutionCandidate, WholeWorkCandidate,
};

use super::{graph::Admitted, LocalCpuAdmission, LocalMultiPlacementRequest};

#[derive(Debug)]
pub(super) enum PlacementHostBoundError {
    UnsupportedRoute,
    Overflow,
}

impl std::fmt::Display for PlacementHostBoundError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedRoute => formatter
                .write_str("finite placement has no proved bounded unpriced physical route"),
            Self::Overflow => formatter.write_str("finite placement host metadata size overflows"),
        }
    }
}

impl std::error::Error for PlacementHostBoundError {}

#[derive(Default)]
struct RequestedBytes(u64);

impl RequestedBytes {
    fn add(&mut self, bytes: usize) -> Result<(), PlacementHostBoundError> {
        let bytes = u64::try_from(bytes).map_err(|_| PlacementHostBoundError::Overflow)?;
        self.0 = self
            .0
            .checked_add(bytes)
            .ok_or(PlacementHostBoundError::Overflow)?;
        Ok(())
    }

    fn vector<T>(&mut self, count: usize) -> Result<(), PlacementHostBoundError> {
        self.add(
            count
                .checked_mul(size_of::<T>())
                .ok_or(PlacementHostBoundError::Overflow)?,
        )
    }
}

/// The generic placement owner can certify one source-bound provider and an
/// optional original VM route without asking a builtin-specific scheduler to
/// recreate its decision. Source/output/program vectors already belong to
/// the caller's disjoint preparation receipt.
pub(super) fn bounded_unpriced_requested_bytes(
    provider: &dyn AccelProvider,
    request: &LocalMultiPlacementRequest<'_>,
) -> Result<u64, PlacementHostBoundError> {
    let [candidate] = request.physical.as_slice() else {
        return Err(PlacementHostBoundError::UnsupportedRoute);
    };
    let has_original = matches!(&request.cpu_admission, LocalCpuAdmission::VmRegionAtUse)
        && request.cpu.route.kind == ExecutionCandidateKind::OriginalVmRegion
        && matches!(
            request.observation_contract,
            super::TotalCandidateObservationContract::FullCall
        )
        && request.vm_right.is_some();
    let provider_only = matches!(&request.cpu_admission, LocalCpuAdmission::Unavailable)
        && matches!(
            request.observation_contract,
            super::TotalCandidateObservationContract::PreparationBodyCleanup
        )
        && request.vm_right.is_none();
    if !(has_original || provider_only)
        || candidate.authority.is_none()
        || candidate.fallback.estimate().is_some()
        || candidate.preparation_program.is_some()
        || candidate.shader_requirements.is_some()
        || candidate.data_inputs.len() != 1
        || candidate
            .source_inputs
            .as_ref()
            .is_none_or(|inputs| inputs.len() != 1)
        || candidate
            .prepared_inputs
            .as_ref()
            .is_none_or(|inputs| inputs.len() != 1)
        || PlacementOperationId::new(request.operation).is_err()
        || provider.capability_snapshot_ref().is_none()
    {
        return Err(PlacementHostBoundError::UnsupportedRoute);
    }
    let mut bytes = RequestedBytes::default();
    // Source route admission, accepted/rejected records and the exact one
    // provider resource snapshot are live together before selection.
    bytes.vector::<Option<ProviderPreparationRoute>>(1)?;
    bytes.vector::<Admitted>(1)?;
    bytes.vector::<(
        Option<runmat_execution::Digest>,
        ProviderFeasibilityQuery,
        runmat_accelerate_api::ProviderRejectionCode,
    )>(1)?;
    bytes.vector::<ProviderResourceSnapshot>(1)?;
    // The source transcript digest uses one borrowed five-field record. No
    // representation vectors are cloned merely to hash this record.
    bytes.vector::<(
        Option<runmat_execution::Digest>,
        Option<&Vec<runmat_accelerate_api::ProviderRepresentation>>,
        Option<&Vec<runmat_accelerate_api::ProviderRepresentation>>,
        Option<ProviderPreparationRoute>,
        Option<u64>,
    )>(1)?;
    // Source revalidation collects one representation. Its constructor first
    // clones the input shape as usize and then collects the u64 wire shape;
    // both vectors can be live together with the outer representation Vec.
    let source_rank = candidate
        .source_inputs
        .as_ref()
        .and_then(|inputs| inputs.first())
        .map(|input| input.shape.len())
        .ok_or(PlacementHostBoundError::UnsupportedRoute)?;
    bytes.vector::<runmat_accelerate_api::ProviderRepresentation>(1)?;
    bytes.vector::<usize>(source_rank)?;
    bytes.vector::<u64>(source_rank)?;
    // The unpriced policy holds its original and filtered candidate vectors
    // simultaneously, plus one compact admitted record in the selector.
    bytes.vector::<WholeWorkCandidate>(if has_original { 4 } else { 2 })?;
    bytes.add(crate::placement::session::unobserved_admitted_record_bytes())?;
    // Candidate-name formatting and the portable selected result own three
    // strings. A device ID is at most ten decimal bytes; Digest Display has
    // the seven-byte `sha256:` prefix plus 64 hexadecimal bytes.
    let provider_name = 9_usize
        .checked_add(10 + 1)
        .and_then(|len| len.checked_add(request.operation.len()))
        .and_then(|len| len.checked_add(1 + 7 + 64))
        .ok_or(PlacementHostBoundError::Overflow)?;
    // Admitted identity, compact selected-candidate presentation and retained
    // managed feedback each own one candidate spelling.
    bytes.add(provider_name)?;
    bytes.add(provider_name)?;
    bytes.add(provider_name)?;
    // The query, returned query copy, signature and its feedback clone each
    // own the bounded operation name at different live points.
    bytes.add(request.operation.len())?;
    bytes.add(request.operation.len())?;
    bytes.add(request.operation.len())?;
    bytes.add(request.operation.len())?;
    // The optional original route owns a fixed typed identity before this
    // quote; its spelling is shorter than the stage's CPU identity allowance.
    // The final boxes own only their struct headers here. Their query's
    // nested source/output/workload vectors are moved from the input record,
    // not cloned by the current unpriced selector.
    bytes.add(size_of::<SelectedExecutionCandidate>())?;
    bytes.add(size_of::<ProviderFeasibilityQuery>())?;
    if has_original {
        // These two digest helpers retain a 96-byte semantic input and one
        // outer-contract label plus 32 digest bytes while computing the
        // complete-call signature. The arrays are sequential, but summing
        // them is safe with the still-live candidate records.
        bytes.add(96)?;
        bytes.add(b"runmat-local-total-candidate-outer-fence-v3".len() + 32)?;
        if let Some(contributions) = request
            .vm_right
            .and_then(|right| right.held_domain_contributions_for_sizing())
        {
            let owned_copies = contributions
                .len()
                .checked_mul(2)
                .ok_or(PlacementHostBoundError::Overflow)?;
            bytes.vector::<DomainContribution>(owned_copies)?;
            for contribution in contributions {
                let names = contribution
                    .name()
                    .len()
                    .checked_mul(2)
                    .ok_or(PlacementHostBoundError::Overflow)?;
                bytes.add(names)?;
            }
            bytes.add(size_of::<super::super::LocalPlacementFeedback>())?;
            bytes.add(size_of::<
                std::sync::Mutex<Option<runmat_runtime::context::RuntimeManagedWholeWorkSelection>>,
            >())?;
            bytes.add(2 * size_of::<std::sync::atomic::AtomicUsize>())?;
            bytes.add(std::mem::align_of::<
                std::sync::Mutex<Option<runmat_runtime::context::RuntimeManagedWholeWorkSelection>>,
            >())?;
        }
    }
    Ok(bytes.0)
}

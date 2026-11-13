//! Sealed Sigma source calls joined to shared ordinary physical recipes.
//! This held owner does not admit full-image resources or numerical equivalence.
use super::*;
use crate::fusion::image_reduction::ImageSelectedSigmaOperations;
use runmat_accelerate_api::{
    ProviderImageMomentProgram, ProviderRealElementwiseObserved, ProviderRealElementwiseRecipe,
};
use runmat_execution::Digest;
use std::cell::{Cell, RefCell};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SelectedSigmaReceipt {
    pub source_input: u64,
    pub source_output: u64,
    pub selected_identity: Digest,
    pub recipe_identity: Digest,
    pub observed: ProviderRealElementwiseObserved,
}

pub(crate) struct SelectedSigmaOperationsPlan {
    source: ImageSelectedSigmaOperations,
    moments: ProviderImageMomentProgram,
    identity: Digest,
    add: ProviderRealElementwiseRecipe,
    sqrt: ProviderRealElementwiseRecipe,
    add_identity: Digest,
    sqrt_identity: Digest,
    receipts: RefCell<Vec<SelectedSigmaReceipt>>,
    cutover: Cell<SigmaCutover>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SigmaCutover {
    AddReady,
    SqrtReady,
    Terminal,
}

#[derive(Clone, Copy)]
enum SigmaOperation {
    Add,
    Sqrt,
}

pub(super) fn plan(
    provider: &'static dyn AccelProvider,
    source: &ImageSelectedSigmaOperations,
    publication: &ordered_mean::SelectedSigmaPublicationPlan,
    moments: &ProviderImageMomentProgram,
    bindings: &ImageSelectedMomentBindings,
) -> ProviderExecutionResult<Option<SelectedSigmaOperationsPlan>> {
    let ids = source.source_ids();
    if source.semantic_identity() != bindings.semantic_identity()
        || !source.matches_moments(moments)
        || publication.raw_source_identity() != (ids.2, ids.3, source.semantic_identity())
        || moments.variance.last().map(|call| call.public_output) != Some(ids.0)
        || (moments.sigma.source_input, moments.sigma.raw_output) != (ids.2, ids.3)
    {
        return Ok(None);
    }
    // This carrier comes from the canonical final variance reduction/cast plan,
    // never from a semantic fact or fabricated future GPU handle.
    let (carrier, bytes) = publication.variance_carrier();
    let Some(add_request) = source.add_request(carrier, bytes) else {
        return Ok(None);
    };
    let Some(add) = ProviderRealElementwiseRecipe::plan(provider, add_request)? else {
        return Ok(None);
    };
    let Some(sqrt_request) = source.sqrt_request(carrier, add.physical().output_buffer_bytes)
    else {
        return Ok(None);
    };
    let Some(sqrt) = ProviderRealElementwiseRecipe::plan(provider, sqrt_request)? else {
        return Ok(None);
    };
    if sqrt.physical().output_buffer_bytes != publication.raw_output_bytes()
        || add.physical().output_buffer_bytes != publication.raw_output_bytes()
        || add
            .physical()
            .requested_incremental_peak_bytes()
            .and_then(|bytes| {
                bytes.checked_add(sqrt.physical().requested_incremental_peak_bytes()?)
            })
            .is_none()
        || add
            .physical()
            .queue_submissions
            .checked_add(sqrt.physical().queue_submissions)
            .is_none()
    {
        return Ok(None);
    }
    let mut receipts = Vec::new();
    receipts
        .try_reserve_exact(2)
        .map_err(ProviderExecutionError::from_source)?;
    Ok(Some(SelectedSigmaOperationsPlan {
        source: source.clone(),
        moments: moments.clone(),
        identity: source.identity().map_err(ProviderExecutionError::from)?,
        add_identity: add.identity().map_err(ProviderExecutionError::from)?,
        sqrt_identity: sqrt.identity().map_err(ProviderExecutionError::from)?,
        add,
        sqrt,
        receipts: RefCell::new(receipts),
        cutover: Cell::new(SigmaCutover::AddReady),
    }))
}

impl SelectedSigmaOperationsPlan {
    pub(super) fn source_ids(&self) -> (u64, u64, u64, u64) {
        self.source.source_ids()
    }
    #[cfg(test)]
    pub(super) fn identity(&self) -> Digest {
        self.identity
    }
    #[cfg(test)]
    pub(super) fn receipts(&self) -> Vec<SelectedSigmaReceipt> {
        self.receipts.borrow().clone()
    }
    pub(super) fn proposals(
        &self,
    ) -> (
        &runmat_accelerate_api::ProviderRealElementwisePhysicalProposal,
        &runmat_accelerate_api::ProviderRealElementwisePhysicalProposal,
    ) {
        (self.add.physical(), self.sqrt.physical())
    }
    #[cfg(test)]
    pub(super) fn recipe_identities(&self) -> (Digest, Digest) {
        (self.add_identity, self.sqrt_identity)
    }

    pub(super) fn matches_program(&self, program: &ProviderImageStageWorkload) -> bool {
        program.moment_program.as_ref() == Some(&self.moments)
            && program.selected_powers.is_some()
            && self.source.control() == program.epsilon
    }

    pub(super) async fn execute_add(
        &self,
        input: &GpuTensorHandle,
        protected: &[&GpuTensorHandle],
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.execute(input, protected, SigmaOperation::Add, admission)
            .await
    }

    pub(super) async fn execute_sqrt(
        &self,
        input: &GpuTensorHandle,
        protected: &[&GpuTensorHandle],
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.execute(input, protected, SigmaOperation::Sqrt, admission)
            .await
    }

    async fn execute(
        &self,
        input: &GpuTensorHandle,
        protected: &[&GpuTensorHandle],
        operation: SigmaOperation,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let ids = self.source_ids();
        let (expected, recipe, recipe_identity, source_input, source_output, completed) =
            match operation {
                SigmaOperation::Add => (
                    SigmaCutover::AddReady,
                    &self.add,
                    self.add_identity,
                    ids.0,
                    ids.2,
                    SigmaCutover::SqrtReady,
                ),
                SigmaOperation::Sqrt => (
                    SigmaCutover::SqrtReady,
                    &self.sqrt,
                    self.sqrt_identity,
                    ids.2,
                    ids.3,
                    SigmaCutover::Terminal,
                ),
            };
        // No duplicate call, reordered call or retry after a successful hook.
        if self.cutover.get() != expected {
            self.cutover.set(SigmaCutover::Terminal);
            return Err(ProviderExecutionError::from(anyhow!(
                "selected Sigma operation is out of order"
            ))
            .after_effect());
        }
        // Cancellation retains a terminal InFlight marker, not a replayable
        // plan. This is set before the first provider future can be polled.
        self.cutover.set(SigmaCutover::Terminal);
        let executed = recipe
            .execute_protected_admitted(input, protected, admission)
            .await
            .map_err(ProviderExecutionError::after_effect)?;
        self.receipts.borrow_mut().push(SelectedSigmaReceipt {
            source_input,
            source_output,
            selected_identity: self.identity,
            recipe_identity,
            observed: executed.observed,
        });
        self.cutover.set(completed);
        Ok(executed.output)
    }
}

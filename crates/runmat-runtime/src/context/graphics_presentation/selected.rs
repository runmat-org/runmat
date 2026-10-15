//! Exact retained scene submitted by the property transaction owner.

use crate::builtins::plotting::state::{
    FigureOwner, FigureSnapshotEntry, GraphicsPropertySelection,
};
use crate::RuntimeError;
use runmat_value::{GraphicsReceiverRef, GraphicsSceneLease};

/// Runtime-only scene authority. Numeric routing coordinates are not issuers.
pub struct RuntimeGraphicsScene {
    pub(crate) receiver: GraphicsReceiverRef,
    pub(crate) snapshot: FigureSnapshotEntry,
}

/// Original scene plus the maximum retention of the actual prepared property
/// prefixes. Descriptive admission evidence, not a new scene or allocation lease.
pub struct RuntimeGraphicsSceneProposal {
    receiver: GraphicsReceiverRef,
    scene: GraphicsSceneLease,
    retained_bytes: usize,
}

impl RuntimeGraphicsSceneProposal {
    pub(crate) fn from_property(
        selection: &GraphicsPropertySelection,
        figure: &FigureOwner,
        payload_bytes: usize,
    ) -> Result<Self, RuntimeError> {
        let snapshot = selection.figure_snapshot_entry(figure).map_err(|source| {
            crate::build_runtime_error("original Figure changed during property preparation")
                .with_identifier("RunMat:graphics:PresentationSceneRetired")
                .with_source(source)
                .build()
        })?;
        let retained_bytes = std::mem::size_of::<GraphicsReceiverRef>()
            .checked_add(
                FigureSnapshotEntry::retained_byte_upper_bound_for_payload(payload_bytes)
                    .ok_or_else(proposal_unavailable)?,
            )
            .ok_or_else(proposal_unavailable)?;
        Ok(Self {
            receiver: selection.figure_receiver().clone(),
            scene: snapshot.scene_lease,
            retained_bytes,
        })
    }
    pub fn receiver(&self) -> &GraphicsReceiverRef {
        &self.receiver
    }
    pub fn scene(&self) -> &GraphicsSceneLease {
        &self.scene
    }
    pub fn retained_byte_upper_bound(&self) -> usize {
        self.retained_bytes
    }
    #[cfg(any(all(feature = "gui", not(target_arch = "wasm32")), test))]
    pub(crate) fn validate_retained_budget(
        &self,
        maximum: std::num::NonZeroUsize,
    ) -> Result<(), crate::builtins::plotting::state::FigurePersistenceSnapshotError> {
        if self.retained_bytes > maximum.get() {
            return Err(
                crate::builtins::plotting::state::FigurePersistenceSnapshotError::BudgetExceeded {
                    retained_bytes: self.retained_bytes,
                    maximum_bytes: maximum.get(),
                },
            );
        }
        Ok(())
    }
}

fn proposal_unavailable() -> RuntimeError {
    crate::build_runtime_error("proposed Figure snapshot retention cannot be bounded")
        .with_identifier("RunMat:graphics:ProposedSceneBudgetUnavailable")
        .build()
}

impl RuntimeGraphicsScene {
    pub(crate) fn from_property(
        selection: &GraphicsPropertySelection,
        figure: &FigureOwner,
    ) -> Result<Self, RuntimeError> {
        let snapshot = selection.figure_snapshot_entry(figure).map_err(|source| {
            crate::build_runtime_error("selected property scene changed before presentation")
                .with_identifier("RunMat:graphics:PresentationSceneRetired")
                .with_source(source)
                .build()
        })?;
        Ok(Self {
            receiver: selection.figure_receiver().clone(),
            snapshot,
        })
    }

    pub fn receiver(&self) -> &GraphicsReceiverRef {
        &self.receiver
    }
    pub fn scene(&self) -> &GraphicsSceneLease {
        &self.snapshot.scene_lease
    }
    pub fn figure(&self) -> &runmat_plot::plots::Figure {
        self.snapshot.figure.as_ref()
    }

    #[cfg(any(all(feature = "gui", not(target_arch = "wasm32")), test))]
    pub(crate) fn retained_byte_upper_bound(&self) -> Option<usize> {
        std::mem::size_of::<GraphicsReceiverRef>()
            .checked_add(self.snapshot.retained_byte_upper_bound()?)
    }
}

pub(super) fn unavailable() -> RuntimeError {
    crate::build_runtime_error("selected graphics host does not support exact scene updates")
        .with_identifier("RunMat:graphics:ExactPresentationUnavailable")
        .build()
}

pub(crate) async fn complete_selected(
    runtime: &crate::context::RuntimeContext,
    service: &dyn super::RuntimeGraphicsPresentationService,
    scene: RuntimeGraphicsScene,
) -> Result<(), RuntimeError> {
    let visible = scene.figure().visible;
    let operation = service.present_scene_update(scene)?;
    let (receipt, disposition) = super::await_completion(
        runtime,
        operation,
        crate::execution::ExecutionServiceError::into_runtime_error,
        || {
            crate::build_runtime_error("graphics host completion returned no receipt")
                .with_identifier("RunMat:graphics:MissingCompletion")
                .build()
        },
    )
    .await?;
    let correct = receipt.callbacks == super::GraphicsCallbackReceipt::Deferred
        && if visible {
            receipt.presentation == super::GraphicsPresentation::Presented
                && disposition == super::GraphicsFlushDisposition::Published
        } else {
            receipt.presentation == super::GraphicsPresentation::NoFigure
        };
    if !correct {
        return Err(super::attach_disposition(
            crate::build_runtime_error("host did not complete the exact scene presentation")
                .with_identifier("RunMat:graphics:InvalidPresentationReceipt")
                .build(),
            disposition,
        ));
    }
    Ok(())
}

//! Linked-axis property notifications retain each actual transaction output.

use super::super::*;

pub(super) enum PreparedFigureUpdates {
    CompatibilityAlias(Vec<(FigureHandle, Figure)>),
    Selected(Vec<SelectedFigureUpdate>),
}

pub(super) struct SelectedFigureUpdate {
    handle: FigureHandle,
    figure: Figure,
    receiver: GraphicsReceiverRef,
    scene: GraphicsSceneLease,
    revision: u64,
}

impl PreparedFigureUpdates {
    pub(super) fn capture(
        property: &PropertyMutationTarget,
        registry: &PlotRegistry,
        updates: Vec<(FigureHandle, Figure)>,
    ) -> Result<Self, FigureError> {
        if !property.is_selected() && property.callback.is_none() {
            return Ok(Self::CompatibilityAlias(updates));
        }
        let selected = updates
            .into_iter()
            .map(|(handle, figure)| {
                let state = registry
                    .figures
                    .get(&handle)
                    .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
                Ok(SelectedFigureUpdate {
                    handle,
                    figure,
                    receiver: state
                        .figure_receiver
                        .clone()
                        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?,
                    scene: state.scene_lease.clone(),
                    revision: state.revision,
                })
            })
            .collect::<Result<Vec<_>, FigureError>>()?;
        Ok(Self::Selected(selected))
    }

    pub(super) fn publish(self, property: &PropertyMutationTarget) -> Result<(), FigureError> {
        match self {
            Self::CompatibilityAlias(updates) => notify_figure_updates(updates),
            Self::Selected(updates) => {
                for update in updates {
                    property.ensure_current()?;
                    {
                        let mut registry = registry();
                        let state = registry
                            .figures
                            .get(&update.handle)
                            .ok_or(FigureError::InvalidHandle(update.handle.as_u32()))?;
                        if !state
                            .figure_receiver
                            .as_ref()
                            .is_some_and(|receiver| receiver.shares_lease(&update.receiver))
                            || !state.scene_lease.shares_lease(&update.scene)
                        {
                            return Err(FigureError::StalePlotSnapshot);
                        }
                        super::super::colorbar::reconcile(
                            &mut registry,
                            update.handle,
                            &update.figure,
                        );
                    }
                    notify_event(FigureEventView {
                        handle: update.handle,
                        kind: FigureEventKind::Updated,
                        revision: Some(update.revision),
                        figure: Some(&update.figure),
                        receiver: Some(&update.receiver),
                        scene: Some(&update.scene),
                    });
                }
            }
        }
        Ok(())
    }
}

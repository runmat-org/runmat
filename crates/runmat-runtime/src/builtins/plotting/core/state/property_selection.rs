//! Exact registry selection retained across property preparation and readback.
//! Numeric coordinates locate an entry only after its original leases validate.

use super::*;

#[derive(Clone, Debug)]
pub(crate) struct GraphicsPropertySelection {
    receiver: GraphicsReceiverRef,
    binding: PropertyBinding,
    figure: FigureHandle,
    figure_receiver: GraphicsReceiverRef,
    scene: GraphicsSceneLease,
}

#[derive(Clone, Debug)]
enum PropertyBinding {
    Figure,
    Axes,
    Child(Arc<PlotChildHandleState>),
}

/// An immutable source snapshot and the exact authority that selected it.
/// The source retains its existing host/device leases; no registry lock awaits.
pub(crate) struct GraphicsPropertySnapshot {
    pub(crate) child: Arc<PlotChildHandleState>,
    pub(crate) figure: super::shared_figure::FigureOwner,
    pub(crate) selection: GraphicsPropertySelection,
}

impl GraphicsPropertySelection {
    pub(crate) fn capture(receiver: GraphicsReceiverRef) -> Result<Self, FigureError> {
        let registry = registry();
        Self::capture_in_registry(receiver, &registry)
    }

    pub(super) fn capture_in_registry(
        receiver: GraphicsReceiverRef,
        registry: &PlotRegistry,
    ) -> Result<Self, FigureError> {
        let figure = match receiver.kind() {
            GraphicsReceiverKind::Figure { figure } | GraphicsReceiverKind::Axes { figure, .. } => {
                FigureHandle::from(figure)
            }
            GraphicsReceiverKind::PlotChild { id } => {
                registry
                    .plot_children
                    .entries
                    .get(&id)
                    .ok_or(FigureError::InvalidPlotObjectHandle)?
                    .state
                    .figure_axes()
                    .0
            }
        };
        let state = registry
            .figures
            .get(&figure)
            .ok_or(FigureError::InvalidHandle(figure.as_u32()))?;
        let binding = match receiver.kind() {
            GraphicsReceiverKind::Figure { .. } => PropertyBinding::Figure,
            GraphicsReceiverKind::Axes { .. } => PropertyBinding::Axes,
            GraphicsReceiverKind::PlotChild { id } => PropertyBinding::Child(
                registry
                    .plot_children
                    .entries
                    .get(&id)
                    .ok_or(FigureError::InvalidPlotObjectHandle)?
                    .state
                    .clone(),
            ),
        };
        let selection = Self {
            receiver,
            binding,
            figure,
            figure_receiver: state
                .figure_receiver
                .clone()
                .ok_or(FigureError::InvalidHandle(figure.as_u32()))?,
            scene: state.scene_lease.clone(),
        };
        selection.validate(registry)?;
        Ok(selection)
    }

    pub(crate) fn receiver(&self) -> &GraphicsReceiverRef {
        &self.receiver
    }

    pub(crate) fn figure_receiver(&self) -> &GraphicsReceiverRef {
        &self.figure_receiver
    }

    pub(crate) fn figure_handle(&self) -> FigureHandle {
        self.figure
    }

    pub(super) fn validate_location(
        &self,
        location: super::property_mutation::PropertyLocation,
    ) -> Result<(), FigureError> {
        use super::property_mutation::PropertyLocation;
        let matches = match location {
            PropertyLocation::CurrentFigure => false,
            PropertyLocation::Annotation(figure, axes, annotation) => {
                figure == self.figure
                    && match &self.binding {
                        PropertyBinding::Figure => true,
                        PropertyBinding::Axes => {
                            matches!(self.receiver.kind(), GraphicsReceiverKind::Axes { index, .. } if index == axes)
                        }
                        PropertyBinding::Child(child) => {
                            matches!(child.as_ref(), PlotChildHandleState::Text(text) if text.axes_index == axes && text.annotation_index == annotation)
                        }
                    }
            }
            PropertyLocation::Figure(figure) => figure == self.figure,
            PropertyLocation::Axes(figure, index) => {
                figure == self.figure
                    && match (&self.binding, self.receiver.kind()) {
                        (PropertyBinding::Figure, _) => true,
                        (
                            PropertyBinding::Axes,
                            GraphicsReceiverKind::Axes {
                                index: selected, ..
                            },
                        ) => index == selected,
                        (PropertyBinding::Child(child), _) => child.figure_axes().1 == index,
                        _ => false,
                    }
            }
            PropertyLocation::Plot(figure, index) => {
                figure == self.figure
                    && matches!(&self.binding, PropertyBinding::Child(child) if child.plot_index() == Some(index))
            }
            PropertyLocation::Child(id) => {
                matches!(self.receiver.kind(), GraphicsReceiverKind::PlotChild { id: selected } if selected == id)
            }
        };
        matches
            .then_some(())
            .ok_or(FigureError::InvalidPlotObjectHandle)
    }

    pub(crate) fn ensure_current(&self) -> Result<(), FigureError> {
        self.validate(&registry()).map(|_| ())
    }

    pub(super) fn successor(&self, registry: &PlotRegistry) -> Result<Self, FigureError> {
        let state = registry
            .figures
            .get(&self.figure)
            .ok_or(FigureError::InvalidHandle(self.figure.as_u32()))?;
        let mut next = self.clone();
        next.scene = state.scene_lease.clone();
        if let GraphicsReceiverKind::PlotChild { id } = self.receiver.kind() {
            next.binding = PropertyBinding::Child(
                registry
                    .plot_children
                    .entries
                    .get(&id)
                    .ok_or(FigureError::InvalidPlotObjectHandle)?
                    .state
                    .clone(),
            );
        }
        // Figure and receiver leases are never replaced by their coordinates.
        next.validate(registry)?;
        Ok(next)
    }

    pub(super) fn validate<'a>(
        &self,
        registry: &'a PlotRegistry,
    ) -> Result<&'a FigureState, FigureError> {
        let state = registry
            .figures
            .get(&self.figure)
            .ok_or(FigureError::InvalidHandle(self.figure.as_u32()))?;
        if !state
            .figure_receiver
            .as_ref()
            .is_some_and(|owned| owned.shares_lease(&self.figure_receiver))
        {
            return Err(FigureError::InvalidHandle(self.figure.as_u32()));
        }
        let selected = match (self.receiver.kind(), &self.binding) {
            (GraphicsReceiverKind::Figure { .. }, PropertyBinding::Figure) => {
                self.receiver.shares_lease(&self.figure_receiver)
            }
            (GraphicsReceiverKind::Axes { index, .. }, PropertyBinding::Axes) => {
                index < axes_count(state)
                    && state
                        .axes_receivers
                        .get(&index)
                        .is_some_and(|owned| owned.shares_lease(&self.receiver))
            }
            (GraphicsReceiverKind::PlotChild { id }, PropertyBinding::Child(selected)) => registry
                .plot_children
                .entries
                .get(&id)
                .is_some_and(|entry| {
                    entry.state.figure_axes().0 == self.figure
                        && Arc::ptr_eq(&entry.state, selected)
                        && entry
                            .receiver
                            .as_ref()
                            .is_some_and(|owned| owned.shares_lease(&self.receiver))
                }),
            _ => false,
        };
        if !selected {
            return Err(FigureError::InvalidPlotObjectHandle);
        }
        if !state.scene_lease.shares_lease(&self.scene) {
            return Err(FigureError::StalePlotSnapshot);
        }
        Ok(state)
    }

    pub(crate) fn figure_snapshot_entry(
        &self,
        retained: &FigureOwner,
    ) -> Result<FigureSnapshotEntry, FigureError> {
        let registry = registry();
        let state = self.validate(&registry)?;
        let figure = state.figure.snapshot();
        if !FigureOwner::ptr_eq(&figure, retained) {
            return Err(FigureError::StalePlotSnapshot);
        }
        Ok(FigureSnapshotEntry {
            handle: self.figure,
            revision: state.revision,
            scene_lease: self.scene.clone(),
            figure,
        })
    }

    pub(crate) fn child_snapshot(&self) -> Result<GraphicsPropertySnapshot, FigureError> {
        let registry = registry();
        let state = self.validate(&registry)?;
        let GraphicsReceiverKind::PlotChild { id } = self.receiver.kind() else {
            return Err(FigureError::InvalidPlotObjectHandle);
        };
        let child = &registry
            .plot_children
            .entries
            .get(&id)
            .ok_or(FigureError::InvalidPlotObjectHandle)?
            .state;
        if let Some(index) = child.plot_index() {
            state
                .figure
                .plots()
                .nth(index)
                .ok_or(FigureError::InvalidPlotObjectHandle)?;
        }
        Ok(GraphicsPropertySnapshot {
            child: child.clone(),
            figure: state.figure.snapshot(),
            selection: self.clone(),
        })
    }
}

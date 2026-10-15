//! Persistent interaction properties live with their exact Figure, not a mode handle.

use super::*;
use runmat_value::{GraphicsCallbackKind, Value};

/// Existing numeric aliases and exact mode-object receivers have distinct
/// admission. A receiver is checked inside the owning registry transaction.
pub enum FigureTarget {
    Alias(FigureHandle),
    Receiver(GraphicsReceiverRef),
}

impl From<FigureHandle> for FigureTarget {
    fn from(handle: FigureHandle) -> Self {
        Self::Alias(handle)
    }
}
impl From<GraphicsReceiverRef> for FigureTarget {
    fn from(receiver: GraphicsReceiverRef) -> Self {
        Self::Receiver(receiver)
    }
}

impl FigureTarget {
    /// Presentation coordinates never authorize the subsequent state access.
    pub(super) fn coordinate(&self) -> FigureHandle {
        match self {
            Self::Alias(handle) => *handle,
            Self::Receiver(receiver) => match receiver.kind() {
                GraphicsReceiverKind::Figure { figure } => FigureHandle::from(figure),
                _ => FigureHandle::from(0),
            },
        }
    }
    pub(super) fn validate(&self, registry: &PlotRegistry) -> Result<(), FigureError> {
        match self {
            Self::Alias(_) => Ok(()),
            Self::Receiver(receiver) => {
                let GraphicsReceiverKind::Figure { figure } = receiver.kind() else {
                    return Err(FigureError::InvalidHandle(0));
                };
                registry
                    .figures
                    .get(&FigureHandle::from(figure))
                    .filter(|state| {
                        state
                            .figure_receiver
                            .as_ref()
                            .is_some_and(|owned| owned.shares_lease(receiver))
                    })
                    .map(|_| ())
                    .ok_or(FigureError::InvalidHandle(figure))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModeProperty {
    Callback(GraphicsCallbackKind),
    PanContextMenu,
    ZoomContextMenu,
    ZoomLegacyContextMenu,
    DataCursorInterpreter,
}

pub(super) struct ModePropertyValue {
    token: callbacks::PlotCallableToken,
}

pub(crate) fn set(
    receiver: &GraphicsReceiverRef,
    key: ModeProperty,
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    PreparedReplacement::new(receiver, key, value, builtin)?.commit()
}

/// Owns the new root until the exact receiver's slot commits. Preparation
/// cannot retire an existing property; failed admission leaves it unchanged.
struct PreparedReplacement {
    receiver: GraphicsReceiverRef,
    handle: FigureHandle,
    key: ModeProperty,
    value: ModePropertyValue,
    builtin: &'static str,
}

impl PreparedReplacement {
    fn new(
        receiver: &GraphicsReceiverRef,
        key: ModeProperty,
        value: Value,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        let handle = figure(receiver, builtin)?;
        let token = if matches!(key, ModeProperty::Callback(_)) {
            callbacks::retain_behavior(value, builtin)?
        } else {
            callbacks::retain(value, builtin)?
        };
        Ok(Self {
            receiver: receiver.clone(),
            handle,
            key,
            value: ModePropertyValue { token },
            builtin,
        })
    }

    fn commit(self) -> BuiltinResult<()> {
        let Self {
            receiver,
            handle,
            key,
            value,
            builtin,
        } = self;
        let old = {
            let mut reg = registry();
            let state = reg
                .figures
                .get_mut(&handle)
                .filter(|state| {
                    state
                        .figure_receiver
                        .as_ref()
                        .is_some_and(|owned| owned.shares_lease(&receiver))
                })
                .ok_or_else(|| super::super::plotting_error(builtin, "figure has retired"))?;
            if let Some((_, previous)) = state
                .mode_properties
                .iter_mut()
                .find(|(property, _)| *property == key)
            {
                Some(std::mem::replace(previous, value))
            } else {
                state.mode_properties.push((key, value));
                None
            }
        };
        // Original root retirement may reenter the owning session.
        drop(old);
        Ok(())
    }
}

pub(crate) fn receiver(
    handle: FigureHandle,
    builtin: &'static str,
) -> BuiltinResult<GraphicsReceiverRef> {
    figure_graphics_receiver(handle)
        .map_err(|cause| {
            super::super::plotting_error_with_source(
                builtin,
                "graphics receiver unavailable",
                cause,
            )
        })?
        .ok_or_else(|| super::super::plotting_error(builtin, "figure has retired"))
}

pub(crate) fn figure(
    receiver: &GraphicsReceiverRef,
    builtin: &'static str,
) -> BuiltinResult<FigureHandle> {
    if !graphics_receiver_valid(receiver) {
        return Err(super::super::plotting_error(
            builtin,
            "mode object's figure has retired",
        ));
    }
    match receiver.kind() {
        GraphicsReceiverKind::Figure { figure } => Ok(FigureHandle::from(figure)),
        _ => Err(super::super::plotting_error(
            builtin,
            "mode object requires an exact Figure receiver",
        )),
    }
}

pub(crate) fn get(
    receiver: &GraphicsReceiverRef,
    key: ModeProperty,
    default: Value,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let handle = figure(receiver, builtin)?;
    let token = {
        let reg = registry();
        let state = reg
            .figures
            .get(&handle)
            .filter(|state| {
                state
                    .figure_receiver
                    .as_ref()
                    .is_some_and(|owned| owned.shares_lease(receiver))
            })
            .ok_or_else(|| super::super::plotting_error(builtin, "figure has retired"))?;
        state
            .mode_properties
            .iter()
            .find(|(property, _)| *property == key)
            .map(|(_, value)| value.token.clone())
    };
    token
        .as_ref()
        .map_or(Ok(default), |token| callbacks::resolve(token, builtin))
}

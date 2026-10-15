//! Persistent interaction properties live with their exact Figure, not a mode handle.

use super::*;
use runmat_value::{GraphicsCallbackKind, Value};
#[path = "mode_properties/responder.rs"]
mod responder;
#[path = "mode_properties/attachment.rs"]
mod attachment;
pub(crate) use attachment::PreparedPropertyAttachment;

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
    ingress: Option<crate::context::GraphicsCallbackIngress>,
}

impl Drop for ModePropertyValue {
    fn drop(&mut self) {
        if let Some(ingress) = &self.ingress {
            ingress.revoke();
        }
    }
}

pub(crate) fn set_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    receiver: &GraphicsReceiverRef,
    key: ModeProperty,
    value: Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    PreparedReplacement::new(runtime, receiver, key, value, builtin)?.commit(runtime)
}

/// Owns the new root until the exact receiver's slot commits. Preparation
/// cannot retire an existing property; failed admission leaves it unchanged.
struct PreparedReplacement {
    receiver: GraphicsReceiverRef,
    handle: FigureHandle,
    key: ModeProperty,
    value: ModePropertyValue,
    predecessor: Option<callbacks::PlotCallableToken>,
    builtin: &'static str,
}

impl PreparedReplacement {
    fn new(
        runtime: Option<&crate::context::RuntimeContext>,
        receiver: &GraphicsReceiverRef,
        key: ModeProperty,
        value: Value,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        let handle = figure(receiver, builtin)?;
        let predecessor = {
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
        let token = if matches!(key, ModeProperty::Callback(_)) {
            match runtime {
                Some(runtime) => {
                    callbacks::retain_behavior_in_context(Some(runtime), value, builtin)?
                }
                None => callbacks::retain_behavior(value, builtin)?,
            }
        } else {
            callbacks::retain(value, builtin)?
        };
        Ok(Self {
            receiver: receiver.clone(),
            handle,
            key,
            value: ModePropertyValue {
                token,
                ingress: None,
            },
            predecessor,
            builtin,
        })
    }

    fn commit(self, runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<()> {
        if let (ModeProperty::Callback(kind), Some(runtime)) = (self.key, runtime) {
            if let Some(queue) = runtime
                .service_ports()
                .graphics_callback_queue()
                .and_then(|queue| queue.upgrade())
            {
                if let Some((selected, callback)) =
                    responder::prepare(&self.value.token, runtime, &self.receiver, kind)?
                {
                    let receiver = self.receiver.clone();
                    let old = queue.replace_selected_property_transaction(
                        selected,
                        receiver,
                        kind,
                        callback,
                        |ingress| self.publish(Some(ingress)),
                    )?;
                    drop(old);
                    return Ok(());
                }
                let receiver = self.receiver.clone();
                let old = self.publish(None)?;
                queue.clear_graphics_property_response(&receiver, kind);
                drop(old);
                return Ok(());
            }
        }
        drop(self.publish(None)?);
        Ok(())
    }

    // This sealed publisher performs only validation and a property-slot move.
    // It returns displaced roots to the queue transaction's outer drop owner.
    fn publish(
        mut self,
        ingress: Option<crate::context::GraphicsCallbackIngress>,
    ) -> BuiltinResult<Option<ModePropertyValue>> {
        self.value.ingress = ingress;
        let Self {
            receiver,
            handle,
            key,
            value,
            predecessor,
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
            let current = state
                .mode_properties
                .iter()
                .find(|(property, _)| *property == key)
                .map(|(_, value)| &value.token);
            if !match (predecessor.as_ref(), current) {
                (None, None) => true,
                (Some(before), Some(current)) => before.shares_lease(current),
                _ => false,
            } {
                return Err(super::super::plotting_error(
                    builtin,
                    "graphics property changed during preparation",
                ));
            }
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
        Ok(old)
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

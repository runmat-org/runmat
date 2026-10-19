//! Prepare original property authorities before publishing a new GUI attachment.

use super::*;
use crate::context::{GraphicsCallbackIngress, RuntimeContext, RuntimeGraphicsCallbackQueue};

struct PreparedProperty {
    receiver: GraphicsReceiverRef,
    key: ModeProperty,
    token: callbacks::PlotCallableToken,
    ingress: GraphicsCallbackIngress,
}

pub(crate) struct PreparedPropertyAttachment(Vec<PreparedProperty>);

impl PreparedPropertyAttachment {
    pub(crate) fn prepare(
        runtime: &RuntimeContext,
        queue: &RuntimeGraphicsCallbackQueue,
        maximum: usize,
    ) -> BuiltinResult<Self> {
        let originals = {
            let reg = registry();
            let mut originals = Vec::new();
            for state in reg.figures.values() {
                let Some(receiver) = &state.figure_receiver else {
                    continue;
                };
                for (key, property) in &state.mode_properties {
                    let ModeProperty::Callback(kind) = key else {
                        continue;
                    };
                    if *kind == GraphicsCallbackKind::DataCursorUpdate
                        || !property.token.belongs_to(runtime)
                    {
                        continue;
                    }
                    if originals.len() >= maximum {
                        return Err(crate::build_runtime_error(
                            "graphics callback registration capacity exhausted",
                        )
                        .with_identifier("RunMat:graphics:CallbackRegistrationCapacity")
                        .build());
                    }
                    originals.push((receiver.clone(), *key, property.token.clone()));
                }
            }
            originals
        };
        let mut prepared = Vec::new();
        for (receiver, key, token) in originals {
            let ModeProperty::Callback(kind) = key else {
                unreachable!("callback-only projection")
            };
            let Some((selected, responder)) = responder::prepare(&token, runtime, &receiver, kind)?
            else {
                continue;
            };
            let ingress = queue.replace_selected_graphics_bound_property_response(
                selected,
                receiver.clone(),
                kind,
                responder,
            )?;
            prepared.push(PreparedProperty {
                receiver,
                key,
                token,
                ingress,
            });
        }
        Ok(Self(prepared))
    }

    pub(crate) fn commit(self) -> BuiltinResult<()> {
        let retired = {
            let mut reg = registry();
            // Validate every original before changing any property ingress.
            for property in &self.0 {
                FigureTarget::Receiver(property.receiver.clone())
                    .validate(&reg)
                    .map_err(|source| {
                        super::super::super::plotting_error_with_source(
                            "graphics",
                            "callback Figure retired during attachment",
                            source,
                        )
                    })?;
                let handle = FigureHandle::from(property.ingress.figure());
                let current = reg
                    .figures
                    .get(&handle)
                    .and_then(|state| {
                        state
                            .mode_properties
                            .iter()
                            .find(|(key, _)| *key == property.key)
                    })
                    .map(|(_, value)| &value.token);
                if !current.is_some_and(|token| token.shares_lease(&property.token)) {
                    return Err(super::super::super::plotting_error(
                        "graphics",
                        "callback property changed during attachment",
                    ));
                }
            }
            let mut retired = Vec::new();
            for property in self.0 {
                let state = reg
                    .figures
                    .get_mut(&FigureHandle::from(property.ingress.figure()))
                    .expect("validated Figure");
                let (_, value) = state
                    .mode_properties
                    .iter_mut()
                    .find(|(key, _)| *key == property.key)
                    .expect("validated property");
                retired.push(value.ingress.replace(property.ingress));
            }
            retired
        };
        for ingress in retired.into_iter().flatten() {
            ingress.revoke();
        }
        Ok(())
    }
}

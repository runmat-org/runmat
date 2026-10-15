//! Exact property admission at the existing get/set boundary.

use super::super::state::{GraphicsPropertySelection, PropertyMutationTarget};
use super::*;
use runmat_value::GraphicsReceiverKind;

pub(super) fn property_unavailable(builtin: &'static str) -> crate::RuntimeError {
    crate::build_runtime_error(
        "graphics object property access requires an exact presentation adapter",
    )
    .with_builtin(builtin)
    .with_identifier("RunMat:graphics:ReceiverPropertyUnavailable")
    .build()
}

fn select(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<Option<GraphicsPropertySelection>> {
    let Value::HandleObject(handle) = value else {
        return Ok(None);
    };
    let receiver = runmat_gc::gc_with_value(&handle.target, |target| match target {
        Value::Object(object) => object.graphics_receiver().cloned(),
        _ => None,
    })
    .ok()
    .flatten();
    let Some(receiver) = receiver else {
        return Ok(None);
    };
    if !crate::is_handle_valid(handle) {
        return Err(crate::build_runtime_error("graphics object is retired")
            .with_builtin(builtin)
            .with_identifier("RunMat:graphics:ReceiverRetired")
            .build());
    }
    GraphicsPropertySelection::capture(receiver)
        .map(Some)
        .map_err(|error| map_figure_error(builtin, error))
}

pub(crate) async fn get_receiver_properties(
    value: &Value,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let Some(selection) = select(value, builtin)? else {
        return Ok(None);
    };
    let result = match selection.receiver().kind() {
        GraphicsReceiverKind::Figure { .. } => {
            figure_read::get_selected(&selection, property, builtin)?
        }
        GraphicsReceiverKind::Axes { .. } => {
            axes_read::get_selected(&selection, property, builtin)?
        }
        GraphicsReceiverKind::PlotChild { .. } => {
            async_readback::get_selected_child_properties(&selection, property, builtin).await?
        }
    };
    Ok(Some(result))
}

/// The exact Figure outcome must be consumed by the owning presentation adapter,
/// not converted to a dirty-only success or passed to the numeric renderer.
pub(crate) async fn set_selected_properties(
    selection: GraphicsPropertySelection,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<PropertySetOutcome> {
    let kind = selection.receiver().kind();
    let authority = PropertyMutationTarget::selected(selection.clone());
    match kind {
        GraphicsReceiverKind::Figure { .. } => set_properties_for_target(
            &authority,
            PropertySetHandle::Figure(selection.figure_handle()),
            args,
            builtin,
        ),
        GraphicsReceiverKind::Axes { index, .. } => set_properties_for_target(
            &authority,
            PropertySetHandle::Axes(selection.figure_handle(), index),
            args,
            builtin,
        ),
        GraphicsReceiverKind::PlotChild { id } => {
            async_readback::set_selected_child_properties(
                &authority, &selection, id, args, builtin,
            )
            .await?;
            Ok(PropertySetOutcome::Complete)
        }
    }
}

pub(crate) async fn set_receiver_properties(
    value: &Value,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<Option<()>> {
    let Some(selection) = select(value, builtin)? else {
        return Ok(None);
    };
    // This boundary has no exact native presentation consumer yet. Decline
    // before writes; activation integration must consume the retained outcome.
    if matches!(
        selection.receiver().kind(),
        GraphicsReceiverKind::Figure { .. }
    ) {
        return Err(property_unavailable(builtin));
    }
    match set_selected_properties(selection, args, builtin).await? {
        PropertySetOutcome::Complete => Ok(Some(())),
        PropertySetOutcome::Presentation(_) => Err(property_unavailable(builtin)),
    }
}

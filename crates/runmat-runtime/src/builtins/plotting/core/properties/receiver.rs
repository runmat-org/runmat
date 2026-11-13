//! Exact property admission at the existing get/set boundary.

use super::super::state::{GraphicsPropertySelection, PropertyMutationTarget};
use super::set_target::PropertyPresentation;
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

pub(crate) fn select(
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

async fn set_selected_properties_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    selection: GraphicsPropertySelection,
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<PropertySetOutcome> {
    let kind = selection.receiver().kind();
    let authority = PropertyMutationTarget::selected(selection.clone()).with_runtime(runtime);
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

pub(crate) async fn set_receiver_properties_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<Option<()>> {
    let Some(selection) = select(value, builtin)? else {
        return Ok(None);
    };
    let (outcome, presentation) = if matches!(
        selection.receiver().kind(),
        GraphicsReceiverKind::Figure { .. }
    ) {
        let runtime = runtime.ok_or_else(|| property_unavailable(builtin))?;
        let service = runtime
            .service_ports()
            .require_graphics_presentation(builtin)
            .map_err(crate::context::RuntimeCapabilityError::into_runtime_error)?
            .clone();
        let authority =
            PropertyMutationTarget::selected(selection.clone()).with_runtime(Some(runtime));
        let plan =
            figure_plan::PreparedFigureProperties::new(selection.figure_handle(), args, builtin)?;
        let proposal = plan.proposal(&selection, builtin)?;
        service.admit_scene_update(&proposal)?;
        let outcome = plan.apply(&authority, selection.figure_handle(), builtin)?;
        (outcome, Some((runtime, service)))
    } else {
        (
            set_selected_properties_with_runtime(runtime, selection, args, builtin).await?,
            None,
        )
    };
    match outcome {
        PropertySetOutcome::Complete => Ok(Some(())),
        PropertySetOutcome::Presentation(PropertyPresentation::Selected { selection, figure }) => {
            let (runtime, service) = presentation.ok_or_else(|| property_unavailable(builtin))?;
            let scene = crate::context::RuntimeGraphicsScene::from_property(&selection, &figure)?;
            crate::context::complete_selected_graphics(runtime, service.as_ref(), scene).await?;
            Ok(Some(()))
        }
        PropertySetOutcome::Presentation(PropertyPresentation::CompatibilityAlias(_, _)) => {
            Err(property_unavailable(builtin))
        }
    }
}

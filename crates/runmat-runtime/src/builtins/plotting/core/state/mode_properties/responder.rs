//! Runtime-lane callable arguments derive only from authenticated queue payloads.

use super::*;
use crate::context::{
    GraphicsSelectedBoundCallbackResponder, RuntimeContext, SessionSourceEvaluation,
};
use runmat_value::{GraphicsCallbackResult, GraphicsCallbackRolePayload, NumericScalar};
use std::rc::Rc;
use std::sync::Arc;

pub(super) fn prepare(
    token: &callbacks::PlotCallableToken,
    attachment: &RuntimeContext,
    receiver: &GraphicsReceiverRef,
    kind: GraphicsCallbackKind,
) -> BuiltinResult<Option<(RuntimeContext, Rc<GraphicsSelectedBoundCallbackResponder>)>> {
    // DataCursor requires an authentic source-point producer, not screen data.
    if kind == GraphicsCallbackKind::DataCursorUpdate {
        return Ok(None);
    }
    let Some(selected) = token.prepare_behavior(attachment)? else {
        return Ok(None);
    };
    let source = token
        .source_text(&selected)
        .map(|text| {
            SessionSourceEvaluation::prepare(
                &selected,
                Arc::from(text),
                Arc::from("<graphics callback>"),
                usize::from(is_filter(kind)),
            )
        })
        .transpose()?;
    let token = token.clone();
    let receiver = receiver.clone();
    let responder: Rc<GraphicsSelectedBoundCallbackResponder> =
        Rc::new(move |runtime, event, payload| {
            let token = token.clone();
            let source = source.clone();
            let receiver = receiver.clone();
            Box::pin(async move {
                if !callback_payload_valid(&receiver, event.kind, &payload) {
                    return Err(super::super::super::plotting_error(
                        "graphics",
                        "callback payload has retired",
                    ));
                }
                if let Some(source) = source {
                    let result = source.evaluate(runtime).await?;
                    return result_for(event.kind, result.as_slice().first());
                }
                let mut args = arguments(&payload)?;
                let value = token.resolve_in_context(&runtime)?;
                let function = match value {
                    Value::Cell(cell) => {
                        let Some(function) = cell.data().first() else {
                            return Err(super::super::super::plotting_error(
                                "graphics",
                                "callback cell is empty",
                            ));
                        };
                        args.extend(cell.data().iter().skip(1).cloned());
                        function.clone()
                    }
                    function => function,
                };
                let result = crate::call_feval_async_with_outputs_in_context(
                    &runtime,
                    function,
                    &args,
                    usize::from(is_filter(event.kind)),
                )
                .await?;
                result_for(event.kind, Some(result.value()))
            })
        });
    Ok(Some((selected, responder)))
}

fn is_filter(kind: GraphicsCallbackKind) -> bool {
    matches!(
        kind,
        GraphicsCallbackKind::PanButtonDownFilter | GraphicsCallbackKind::ZoomButtonDownFilter
    )
}

fn receiver_value(receiver: &GraphicsReceiverRef) -> BuiltinResult<Value> {
    graphics_receiver_object(receiver.clone())
        .map_err(|source| {
            super::super::super::plotting_error_with_source(
                "graphics",
                "callback receiver unavailable",
                source,
            )
        })?
        .ok_or_else(|| {
            super::super::super::plotting_error("graphics", "callback receiver has retired")
        })
}

fn arguments(payload: &GraphicsCallbackRolePayload) -> BuiltinResult<Vec<Value>> {
    match payload {
        GraphicsCallbackRolePayload::Action { figure, axes, .. } => {
            let mut event = runmat_value::UnboundedStructValueBuilder::new();
            event.insert("Axes", receiver_value(axes)?);
            Ok(vec![receiver_value(figure)?, Value::Struct(event.finish())])
        }
        GraphicsCallbackRolePayload::Filter { target, .. } => {
            let empty = Tensor::new(Vec::new(), vec![0, 0])
                .map_err(|message| super::super::super::plotting_error("graphics", message))?;
            Ok(vec![receiver_value(target)?, Value::Tensor(empty)])
        }
    }
}

fn result_for(
    kind: GraphicsCallbackKind,
    value: Option<&Value>,
) -> BuiltinResult<GraphicsCallbackResult> {
    if !is_filter(kind) {
        return Ok(GraphicsCallbackResult::Completed);
    }
    let numeric = match value {
        Some(Value::Bool(value)) => return Ok(GraphicsCallbackResult::Suppress(*value)),
        Some(Value::LogicalArray(array)) if array.data.len() == 1 => {
            return Ok(GraphicsCallbackResult::Suppress(array.data[0] != 0))
        }
        Some(Value::Num(value)) => Some(NumericScalar::F64(*value)),
        Some(Value::Int(value)) => Some(value.clone().into()),
        Some(Value::Tensor(tensor)) if tensor.len() == 1 => tensor.numeric_value_at(0),
        _ => None,
    };
    match numeric.map(NumericScalar::materialize_f64) {
        Some(0.0) => Ok(GraphicsCallbackResult::Suppress(false)),
        Some(1.0) => Ok(GraphicsCallbackResult::Suppress(true)),
        _ => Err(crate::build_runtime_error(
            "graphics filter must return a scalar logical or numeric zero or one",
        )
        .with_identifier("RunMat:graphics:CallbackResult")
        .build()),
    }
}

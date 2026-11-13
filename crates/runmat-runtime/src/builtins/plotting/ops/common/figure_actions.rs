use crate::builtins::plotting::state::FigureHandle;
use crate::BuiltinResult;

use super::handles::{handle_from_string, handles_from_value, parse_string};

#[derive(Debug)]
pub enum FigureAction {
    Current,
    Handles(Vec<FigureHandle>),
    All,
}

pub fn parse_clf_action(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(FigureAction, bool)> {
    if args.is_empty() {
        return Ok((FigureAction::Current, false));
    }
    let mut handles = Vec::new();
    let mut clear_all = false;
    let mut reset = false;
    for value in args {
        if let Some(text) = crate::builtins::common::validation::value_text_scalar_string(value) {
            let normalized = text.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                continue;
            }
            match normalized.as_str() {
                "all" => {
                    clear_all = true;
                    continue;
                }
                "reset" => {
                    reset = true;
                    continue;
                }
                _ => {
                    handles.push(handle_from_string(&normalized, "clf")?);
                    continue;
                }
            }
        }
        handles.extend(handles_from_value(value, "clf")?);
    }
    if clear_all {
        Ok((FigureAction::All, reset))
    } else if handles.is_empty() {
        Ok((FigureAction::Current, reset))
    } else {
        Ok((FigureAction::Handles(handles), reset))
    }
}

#[cfg(test)]
pub fn parse_close_action(args: &[runmat_value::Value]) -> BuiltinResult<FigureAction> {
    parse_close_action_view(crate::call::FunctionArgumentView::from_slice(args))
}

pub fn parse_close_action_view(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<FigureAction> {
    if args.is_empty() {
        return Ok(FigureAction::Current);
    }
    let mut handles = Vec::new();
    let mut close_all = false;
    for value in args {
        if let Some(text) = parse_string(value).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "close",
                source.to_string(),
                source,
            )
        })? {
            let normalized = text.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                continue;
            }
            if matches!(normalized.as_str(), "all" | "force" | "force all") {
                close_all = true;
                continue;
            }
            if normalized == "current" {
                continue;
            }
            handles.push(handle_from_string(&normalized, "close")?);
            continue;
        }
        handles.extend(handles_from_value(value, "close")?);
    }
    if close_all {
        Ok(FigureAction::All)
    } else if handles.is_empty() {
        Ok(FigureAction::Current)
    } else {
        Ok(FigureAction::Handles(handles))
    }
}

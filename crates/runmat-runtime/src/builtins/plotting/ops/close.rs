//! MATLAB-compatible `close` builtin.

use std::collections::BTreeSet;

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::handle_args::{handle_from_string, handles_from_value, parse_string};
use super::state::{close_figure, figure_handles, FigureHandle};

#[runtime_builtin(
    name = "close",
    category = "plotting",
    summary = "Close figures by handle or the active figure.",
    keywords = "close,figure,plotting",
    sink = true
)]
pub fn close_builtin(rest: Vec<Value>) -> Result<String, String> {
    match parse_close_action(&rest)? {
        CloseAction::Current => {
            let closed = close_figure(None).map_err(|err| format!("close: {err}"))?;
            Ok(format!("Closed figure {}", closed.as_u32()))
        }
        CloseAction::Handles(handles) => {
            let unique: BTreeSet<u32> = handles.into_iter().map(|h| h.as_u32()).collect();
            if unique.is_empty() {
                let closed = close_figure(None).map_err(|err| format!("close: {err}"))?;
                return Ok(format!("Closed figure {}", closed.as_u32()));
            }
            let mut closed = Vec::new();
            for id in unique {
                let handle = FigureHandle::from(id);
                close_figure(Some(handle)).map_err(|err| format!("close: {err}"))?;
                closed.push(id);
            }
            if closed.len() == 1 {
                Ok(format!("Closed figure {}", closed[0]))
            } else {
                Ok(format!("Closed {} figures", closed.len()))
            }
        }
        CloseAction::All => {
            let handles = figure_handles();
            if handles.is_empty() {
                return Ok("close: no figures to close".to_string());
            }
            for handle in handles {
                let _ = close_figure(Some(handle));
            }
            Ok("Closed all figures".to_string())
        }
    }
}

#[derive(Debug)]
enum CloseAction {
    Current,
    Handles(Vec<FigureHandle>),
    All,
}

fn parse_close_action(args: &[Value]) -> Result<CloseAction, String> {
    if args.is_empty() {
        return Ok(CloseAction::Current);
    }
    let mut handles = Vec::new();
    let mut close_all = false;
    for value in args {
        if let Some(text) = parse_string(value) {
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
        return Ok(CloseAction::All);
    }
    if handles.is_empty() {
        Ok(CloseAction::Current)
    } else {
        Ok(CloseAction::Handles(handles))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[ctor::ctor]
    fn init_plot_test_env() {
        crate::builtins::plotting::state::disable_rendering_for_tests();
    }

    #[test]
    fn parse_defaults_to_current() {
        assert!(matches!(
            parse_close_action(&[]).unwrap(),
            CloseAction::Current
        ));
    }

    #[test]
    fn parse_numeric_handles() {
        let values = vec![Value::Num(3.0), Value::Num(1.0)];
        match parse_close_action(&values).unwrap() {
            CloseAction::Handles(handles) => {
                assert_eq!(handles.len(), 2);
                assert_eq!(handles[0].as_u32(), 3);
                assert_eq!(handles[1].as_u32(), 1);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn parse_all_flag() {
        let values = vec![Value::String("all".to_string())];
        assert!(matches!(
            parse_close_action(&values).unwrap(),
            CloseAction::All
        ));
    }
}

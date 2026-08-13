use crate::source_context;
use runmat_thread_local::runmat_thread_local;
use runmat_types::SourceId;
use std::cell::RefCell;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugFrame {
    pub function: String,
    pub source_id: Option<SourceId>,
    pub span: Option<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugFrameInfo {
    pub function: String,
    pub file: String,
    pub line: usize,
}

runmat_thread_local! {
    static DEBUG_STACK: RefCell<Vec<DebugFrame>> = const { RefCell::new(Vec::new()) };
}

pub struct DebugFrameGuard {
    did_push: bool,
}

impl Drop for DebugFrameGuard {
    fn drop(&mut self) {
        if !self.did_push {
            return;
        }
        DEBUG_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let _ = stack.pop();
        });
    }
}

pub fn push_frame(
    function: impl Into<String>,
    source_id: Option<SourceId>,
    span: Option<(usize, usize)>,
) -> DebugFrameGuard {
    DEBUG_STACK.with(|stack| {
        stack.borrow_mut().push(DebugFrame {
            function: function.into(),
            source_id,
            span,
        });
    });
    DebugFrameGuard { did_push: true }
}

pub fn current_frames() -> Vec<DebugFrameInfo> {
    DEBUG_STACK.with(|stack| {
        stack
            .borrow()
            .iter()
            .rev()
            .map(frame_info)
            .collect::<Vec<_>>()
    })
}

pub fn current_function_name() -> Option<String> {
    DEBUG_STACK.with(|stack| stack.borrow().last().map(|frame| frame.function.clone()))
}

pub fn reset_for_tests() {
    DEBUG_STACK.with(|stack| stack.borrow_mut().clear());
}

fn frame_info(frame: &DebugFrame) -> DebugFrameInfo {
    let source = frame.source_id.and_then(source_context::source_info);
    let file = source
        .as_ref()
        .map(|source| {
            source
                .fullpath_name
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| source.name.to_string())
        })
        .unwrap_or_default();
    let line = frame
        .span
        .and_then(|(start, _)| {
            source
                .as_ref()
                .map(|source| line_for_offset(&source.text, start))
        })
        .unwrap_or(0);
    DebugFrameInfo {
        function: frame.function.clone(),
        file,
        line,
    }
}

fn line_for_offset(source: &str, offset: usize) -> usize {
    let offset = offset.min(source.len());
    source[..offset]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_frame_snapshot_uses_source_catalog() {
        let source_id = SourceId(7);
        let _catalog = source_context::replace_source_catalog_with_fullpaths(vec![(
            source_id,
            "demo.m".to_string(),
            Some("/tmp/demo.m".to_string()),
            "a = 1;\nb = 2;\n".to_string(),
        )]);
        let _guard = push_frame("demo", Some(source_id), Some((8, 13)));
        let frames = current_frames();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].function, "demo");
        assert_eq!(frames[0].file, "/tmp/demo.m");
        assert_eq!(frames[0].line, 2);
    }
}

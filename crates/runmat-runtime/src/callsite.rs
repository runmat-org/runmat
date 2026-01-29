use crate::source_context;
use runmat_hir::{SourceId, Span};
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;

#[derive(Debug, Clone)]
pub struct CallsiteInfo {
    pub source_id: Option<SourceId>,
    pub arg_spans: Vec<Span>,
}

runmat_thread_local! {
    static CALLSITE_STACK: RefCell<Vec<CallsiteInfo>> = const { RefCell::new(Vec::new()) };
}

pub struct CallsiteGuard {
    did_push: bool,
}

impl Drop for CallsiteGuard {
    fn drop(&mut self) {
        if !self.did_push {
            return;
        }
        CALLSITE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let _ = stack.pop();
        });
    }
}

pub fn push_callsite(source_id: Option<SourceId>, arg_spans: Option<Vec<Span>>) -> CallsiteGuard {
    let Some(arg_spans) = arg_spans else {
        return CallsiteGuard { did_push: false };
    };
    CALLSITE_STACK.with(|stack| {
        stack.borrow_mut().push(CallsiteInfo {
            source_id,
            arg_spans,
        });
    });
    CallsiteGuard { did_push: true }
}

pub fn current_callsite() -> Option<CallsiteInfo> {
    CALLSITE_STACK.with(|stack| stack.borrow().last().cloned())
}

fn clamp_to_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx > s.len() {
        idx = s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn normalize_label_text(raw: &str) -> String {
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim();
    const MAX_LEN: usize = 80;
    if trimmed.len() <= MAX_LEN {
        return trimmed.to_string();
    }
    // Conservative truncation: keep valid UTF-8 boundaries.
    let mut end = MAX_LEN;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &trimmed[..end])
}

pub fn arg_text(arg_index: usize) -> Option<String> {
    let callsite = current_callsite()?;
    let source = source_context::current_source()?;
    let span = callsite.arg_spans.get(arg_index)?;

    let start = clamp_to_char_boundary(&source, span.start);
    let end = clamp_to_char_boundary(&source, span.end);
    if end <= start {
        return None;
    }
    let raw = &source[start..end];
    let text = normalize_label_text(raw);
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

use crate::source_context;
use runmat_thread_local::runmat_thread_local;
use runmat_types::SourceId;
use std::cell::RefCell;
use std::rc::Rc;

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

#[derive(Debug)]
struct FrameNode {
    frame: DebugFrame,
    parent: Option<Rc<FrameNode>>,
}

/// One invocation's immutable async frames and exact-owner synchronous frames.
/// A poll owner forks this projection; sibling futures never mutate a shared
/// session stack while suspended.
#[derive(Debug, Default)]
pub(crate) struct DebugFrameProjection {
    head: Option<Rc<FrameNode>>,
    overlays: Vec<(u64, DebugFrame)>,
    next_overlay: u64,
}

impl DebugFrameProjection {
    fn linked_head(&self) -> Option<Rc<FrameNode>> {
        let mut head = self.head.clone();
        for (_, frame) in &self.overlays {
            head = Some(Rc::new(FrameNode {
                frame: frame.clone(),
                parent: head,
            }));
        }
        head
    }

    pub(crate) fn snapshot(&self) -> Self {
        Self {
            head: self.linked_head(),
            overlays: Vec::new(),
            next_overlay: 0,
        }
    }

    pub(crate) fn with_frame(&self, frame: DebugFrame) -> Self {
        Self {
            head: Some(Rc::new(FrameNode {
                frame,
                parent: self.linked_head(),
            })),
            overlays: Vec::new(),
            next_overlay: 0,
        }
    }

    fn push(&mut self, frame: DebugFrame) -> u64 {
        let token = self.next_overlay;
        self.next_overlay = self
            .next_overlay
            .checked_add(1)
            .expect("debug frame token space exhausted");
        self.overlays.push((token, frame));
        token
    }

    fn remove(&mut self, token: u64) {
        if let Some(index) = self.overlays.iter().position(|(entry, _)| *entry == token) {
            self.overlays.remove(index);
        }
    }

    fn current_function_name(&self) -> Option<String> {
        self.overlays
            .last()
            .map(|(_, frame)| frame.function.clone())
            .or_else(|| self.head.as_ref().map(|node| node.frame.function.clone()))
    }

    fn frames(&self) -> Vec<DebugFrame> {
        let mut frames = self
            .overlays
            .iter()
            .rev()
            .map(|(_, frame)| frame.clone())
            .collect::<Vec<_>>();
        let mut current = self.head.as_deref();
        while let Some(node) = current {
            frames.push(node.frame.clone());
            current = node.parent.as_deref();
        }
        frames
    }

    fn clear(&mut self) {
        self.head = None;
        self.overlays.clear();
    }
}

runmat_thread_local! {
    static DIRECT_FRAMES: RefCell<DebugFrameProjection> = RefCell::new(DebugFrameProjection::default());
}

enum GuardOwner {
    Selected(Rc<RefCell<DebugFrameProjection>>),
    Direct,
}

pub struct DebugFrameGuard {
    owner: GuardOwner,
    token: u64,
}

impl Drop for DebugFrameGuard {
    fn drop(&mut self) {
        match &self.owner {
            GuardOwner::Selected(projection) => projection.borrow_mut().remove(self.token),
            GuardOwner::Direct => {
                DIRECT_FRAMES.with(|frames| frames.borrow_mut().remove(self.token))
            }
        }
    }
}

/// Synchronous embedding frame. Async VM calls use `RuntimeContext::with_debug_frame`
/// so a suspension never leaves an ambient frame installed on the thread.
pub fn push_frame(
    function: impl Into<String>,
    source_id: Option<SourceId>,
    span: Option<(usize, usize)>,
) -> DebugFrameGuard {
    let frame = DebugFrame {
        function: function.into(),
        source_id,
        span,
    };
    if let Some(context) = crate::context::legacy::active() {
        let projection = Rc::clone(context.debug_projection());
        let token = projection.borrow_mut().push(frame);
        DebugFrameGuard {
            owner: GuardOwner::Selected(projection),
            token,
        }
    } else {
        let token = DIRECT_FRAMES.with(|frames| frames.borrow_mut().push(frame));
        DebugFrameGuard {
            owner: GuardOwner::Direct,
            token,
        }
    }
}

pub fn current_frames() -> Vec<DebugFrameInfo> {
    let frames = if let Some(context) = crate::context::legacy::active() {
        context.debug_projection().borrow().frames()
    } else {
        DIRECT_FRAMES.with(|frames| frames.borrow().frames())
    };
    frames.iter().map(frame_info).collect()
}

pub fn current_function_name() -> Option<String> {
    if let Some(context) = crate::context::legacy::active() {
        context.debug_projection().borrow().current_function_name()
    } else {
        DIRECT_FRAMES.with(|frames| frames.borrow().current_function_name())
    }
}

pub fn reset_for_tests() {
    if let Some(context) = crate::context::legacy::active() {
        context.debug_projection().borrow_mut().clear();
    } else {
        DIRECT_FRAMES.with(|frames| frames.borrow_mut().clear());
    }
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
#[path = "debug_context/tests.rs"]
mod tests;

use crate::source_context;
use runmat_execution::resource::ExactShared;
use runmat_thread_local::runmat_thread_local;
use runmat_types::SourceId;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};
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

#[derive(Debug)]
struct FrameNode {
    frame: DebugFrame,
    parent: Option<ExactShared<FrameNode>>,
    _header_claim: Option<ExactHostAllocationClaim>,
    _name_claim: Option<ExactHostAllocationClaim>,
}

/// One invocation's immutable async frames and exact-owner synchronous frames.
/// A poll owner forks this projection; sibling futures never mutate a shared
/// session stack while suspended.
#[derive(Debug, Default)]
pub(crate) struct DebugFrameProjection {
    head: Option<ExactShared<FrameNode>>,
    overlays: Vec<(u64, DebugFrame)>,
    next_overlay: u64,
}

impl DebugFrameProjection {
    fn linked_head(&self) -> Option<ExactShared<FrameNode>> {
        let mut head = self.head.clone();
        for (_, frame) in &self.overlays {
            head = Some(ExactShared::new(FrameNode {
                frame: frame.clone(),
                parent: head,
                _header_claim: None,
                _name_claim: None,
            }));
        }
        head
    }

    /// Exact physical allocations made when materializing active synchronous
    /// overlays for one managed poll snapshot. Existing immutable ancestors
    /// are retained without copying their names or nodes.
    pub(crate) fn snapshot_requested_bytes(&self) -> Option<usize> {
        self.overlays.iter().try_fold(0usize, |bytes, (_, frame)| {
            bytes
                .checked_add(ExactShared::<FrameNode>::requested_bytes())?
                .checked_add(frame.function.len())
        })
    }

    pub(crate) fn snapshot_retained_claims(&self) -> Option<usize> {
        self.overlays.len().checked_mul(2)
    }

    pub(crate) fn snapshot_with_permit(
        &self,
        permit: &crate::context::host_allocation::HostOutputPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let mut head = self.head.clone();
        for (_, frame) in &self.overlays {
            let header_claim =
                permit.reserve_exact_claim(ExactShared::<FrameNode>::requested_bytes())?;
            let name_claim = permit.reserve_exact_claim(frame.function.len())?;
            let mut function = String::new();
            function
                .try_reserve_exact(frame.function.len())
                .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
            function.push_str(&frame.function);
            let frame = DebugFrame {
                function,
                source_id: frame.source_id,
                span: frame.span,
            };
            head = Some(
                ExactShared::try_new(FrameNode {
                    frame,
                    parent: head,
                    _header_claim: Some(header_claim),
                    _name_claim: Some(name_claim),
                })
                .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?,
            );
        }
        Ok(Self {
            head,
            overlays: Vec::new(),
            next_overlay: 0,
        })
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
            head: Some(ExactShared::new(FrameNode {
                frame,
                parent: self.linked_head(),
                _header_claim: None,
                _name_claim: None,
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
            .or_else(|| {
                self.head
                    .as_ref()
                    .map(|node| node.get().frame.function.clone())
            })
    }

    fn frames(&self) -> Vec<DebugFrame> {
        let mut frames = self
            .overlays
            .iter()
            .rev()
            .map(|(_, frame)| frame.clone())
            .collect::<Vec<_>>();
        let mut current = self.head.as_ref().map(ExactShared::get);
        while let Some(node) = current {
            frames.push(node.frame.clone());
            current = node.parent.as_ref().map(ExactShared::get);
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
    Selected(crate::context::poll_projection::PollProjection<DebugFrameProjection>),
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
        let projection = context.debug_projection().clone();
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

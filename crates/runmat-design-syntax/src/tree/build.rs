//! Assembles the green tree from the token stream and the nodes the parser recognised.
//!
//! The parser records each construct as a token-index interval, in whatever order it decides
//! the construct (a block member's body is decided before its header, for instance). Assembly
//! sorts the intervals, nests them, and threads every token, trivia included, into the
//! innermost node that covers it. The result holds every byte of the source exactly once.

use super::green::{GreenElement, GreenNode, GreenToken};
use super::kind::NodeKind;
use crate::lexical::FeaTokenStream;

/// One recognised construct: tokens `start..end` of the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeInterval {
    pub kind: NodeKind,
    pub start: usize,
    pub end: usize,
}

/// Records intervals while a parser runs.
#[derive(Debug, Default, Clone)]
pub(crate) struct TreeRecorder {
    intervals: Vec<NodeInterval>,
}

impl TreeRecorder {
    pub fn node(&mut self, kind: NodeKind, start: usize, end: usize) {
        if start < end {
            self.intervals.push(NodeInterval { kind, start, end });
        }
    }

    /// A node over a run of significant token indices (first to last, trivia between).
    pub fn node_over(&mut self, kind: NodeKind, indices: &[usize]) {
        if let (Some(first), Some(last)) = (indices.first(), indices.last()) {
            self.node(kind, *first, *last + 1);
        }
    }

    pub fn finish(self, stream: &FeaTokenStream) -> GreenNode {
        assemble(stream, self.intervals)
    }
}

pub(crate) fn assemble(stream: &FeaTokenStream, mut intervals: Vec<NodeInterval>) -> GreenNode {
    let count = stream.tokens.len();
    intervals.retain(|interval| interval.end <= count && interval.start < interval.end);
    // Outer first: earlier start, then later end, then the enclosing rank.
    let mut ordered: Vec<(usize, NodeInterval)> = intervals.into_iter().enumerate().collect();
    ordered.sort_by(|(left_order, left), (right_order, right)| {
        left.start
            .cmp(&right.start)
            .then(right.end.cmp(&left.end))
            .then(left.kind.rank().cmp(&right.kind.rank()))
            .then(left_order.cmp(right_order))
    });
    ordered.dedup_by(|(_, a), (_, b)| a == b);

    struct Frame {
        kind: NodeKind,
        end: usize,
        children: Vec<GreenElement>,
    }
    let mut stack = vec![Frame {
        kind: NodeKind::SourceFile,
        end: count,
        children: Vec::new(),
    }];
    let mut pending = ordered.into_iter().map(|(_, interval)| interval).peekable();
    for (index, token) in stream.tokens.iter().enumerate() {
        while let Some(interval) = pending.peek().copied() {
            if interval.start > index {
                break;
            }
            pending.next();
            // An interval that starts before this token was crossed by an earlier one, and an
            // interval that ends past its parent crosses it; neither can nest, so neither is
            // kept. Parsers only record well-nested constructs; this guards error recovery.
            let parent_end = stack.last().map_or(count, |frame| frame.end);
            if interval.start < index || interval.end > parent_end {
                continue;
            }
            stack.push(Frame {
                kind: interval.kind,
                end: interval.end,
                children: Vec::new(),
            });
        }
        let text = &stream.source[token.span.start..token.span.end];
        stack
            .last_mut()
            .expect("the root frame is never closed early")
            .children
            .push(GreenElement::Token(GreenToken::new(token.kind, text)));
        while stack.len() > 1 && stack.last().is_some_and(|frame| frame.end == index + 1) {
            let frame = stack.pop().unwrap();
            let node = GreenNode::new(frame.kind, frame.children);
            stack
                .last_mut()
                .unwrap()
                .children
                .push(GreenElement::Node(node));
        }
    }
    while stack.len() > 1 {
        let frame = stack.pop().unwrap();
        let node = GreenNode::new(frame.kind, frame.children);
        stack
            .last_mut()
            .unwrap()
            .children
            .push(GreenElement::Node(node));
    }
    let root = stack.pop().unwrap();
    GreenNode::new(root.kind, root.children)
}

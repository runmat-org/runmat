use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

pub fn merge_span(lhs: Span, rhs: Span) -> Span {
    Span {
        start: lhs.start.min(rhs.start),
        end: lhs.end.max(rhs.end),
    }
}

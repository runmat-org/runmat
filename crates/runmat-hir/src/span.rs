pub type Span = runmat_parser::Span;

pub fn merge_span(lhs: Span, rhs: Span) -> Span {
    Span {
        start: lhs.start.min(rhs.start),
        end: lhs.end.max(rhs.end),
    }
}

/// Returns whether a produced field identifier satisfies an exact or dotted-suffix request.
///
/// Callers remain responsible for exact-match precedence and ambiguity rejection across their
/// candidate set. Keeping the lexical rule here prevents result filtering, artifact paging, and
/// language projection from assigning different meanings to the same request.
pub(crate) fn field_id_matches(candidate: &str, requested: &str) -> bool {
    candidate == requested
        || candidate
            .strip_suffix(requested)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

#[cfg(test)]
mod tests {
    use super::field_id_matches;

    #[test]
    fn matching_requires_an_exact_id_or_complete_dotted_suffix() {
        assert!(field_id_matches(
            "structural.displacement",
            "structural.displacement"
        ));
        assert!(field_id_matches("structural.displacement", "displacement"));
        assert!(!field_id_matches(
            "structural.nodal_displacement",
            "displacement"
        ));
        assert!(!field_id_matches("structural.displacement", "placement"));
        assert!(!field_id_matches("displacement", "structural.displacement"));
    }
}

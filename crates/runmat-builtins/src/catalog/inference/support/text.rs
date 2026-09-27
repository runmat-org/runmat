use runmat_types::ShapeFact;

/// A row of characters, allowing trailing singleton dimensions. Unknown
/// dimensions cannot prove rejection; zero-row empty text retains its policy.
pub(crate) fn possible_character_row(shape: &ShapeFact) -> bool {
    shape.known_dims().is_none_or(|dims| {
        !dims.first().copied().flatten().is_some_and(|rows| rows > 1)
            && !dims
                .iter()
                .skip(2)
                .any(|dim| dim.is_some_and(|size| size != 1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_rows_preserve_unknowns_and_semantic_rank() {
        for shape in [
            vec![Some(1), Some(4)],
            vec![Some(1), Some(4), Some(1)],
            vec![None, Some(4), None],
            vec![Some(0), Some(0)],
        ] {
            assert!(possible_character_row(&ShapeFact::from(shape)));
        }
        for shape in [
            vec![Some(2), Some(2)],
            vec![Some(1), Some(2), Some(2)],
            vec![None, None, Some(2)],
        ] {
            assert!(!possible_character_row(&ShapeFact::from(shape)));
        }
    }
}

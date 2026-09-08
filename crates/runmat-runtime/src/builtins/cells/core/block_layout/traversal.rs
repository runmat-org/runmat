pub(in crate::builtins::cells::core) fn column_major_coordinates(
    mut linear: usize,
    shape: &[usize],
) -> Vec<usize> {
    shape
        .iter()
        .map(|extent| {
            if *extent == 0 {
                0
            } else {
                let coordinate = linear % *extent;
                linear /= *extent;
                coordinate
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visits_first_dimension_fastest() {
        let visited = (0..4)
            .map(|index| column_major_coordinates(index, &[2, 2]))
            .collect::<Vec<_>>();
        assert_eq!(
            visited,
            vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]
        );
    }
}

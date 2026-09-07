pub(in crate::builtins::math::elementwise::binary_arithmetic) fn broadcast_repetitions(
    left: &[usize],
    right: &[usize],
) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rank = left.len().max(right.len()).max(1);
    let left = crate::builtins::common::broadcast::align_shape(left, rank);
    let right = crate::builtins::common::broadcast::align_shape(right, rank);
    let mut output = Vec::with_capacity(rank);
    for (&left_extent, &right_extent) in left.iter().zip(&right) {
        output.push(match (left_extent, right_extent) {
            (left, right) if left == right => left,
            (1, right) => right,
            (left, 1) => left,
            _ => return None,
        });
    }
    Some((
        output.clone(),
        repetitions(&left, &output),
        repetitions(&right, &output),
    ))
}

fn repetitions(shape: &[usize], output: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .zip(output)
        .map(|(&extent, &output_extent)| {
            if extent == output_extent {
                1
            } else {
                output_extent
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::broadcast_repetitions;

    #[test]
    fn aligns_ranks_and_singletons() {
        let (shape, left, right) =
            broadcast_repetitions(&[3, 1], &[1, 4]).expect("compatible shapes");
        assert_eq!(shape, vec![3, 4]);
        assert_eq!(left, vec![1, 4]);
        assert_eq!(right, vec![3, 1]);
        assert!(broadcast_repetitions(&[2, 3], &[4, 3]).is_none());
    }
}

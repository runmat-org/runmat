use anyhow::{anyhow, ensure, Result};

pub(super) fn floating_elements<T: Copy>(
    data: &[T],
    lane_factor: usize,
    source_elements: usize,
    indices: &[u32],
    output_elements: usize,
) -> Result<Vec<T>> {
    ensure!(
        indices.len() == output_elements,
        "gather_linear: index count {} does not match output size {}",
        indices.len(),
        output_elements
    );
    let expected_lanes = source_elements
        .checked_mul(lane_factor)
        .ok_or_else(|| anyhow!("gather_linear: source lane count overflow"))?;
    ensure!(
        data.len() == expected_lanes,
        "gather_linear: source payload has {} lanes, expected {}",
        data.len(),
        expected_lanes
    );
    let output_lanes = output_elements
        .checked_mul(lane_factor)
        .ok_or_else(|| anyhow!("gather_linear: output lane count overflow"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_lanes)
        .map_err(|source| anyhow!(source).context("gather_linear: output allocation failed"))?;
    for (position, &index) in indices.iter().enumerate() {
        let source_index = index as usize;
        ensure!(
            source_index < source_elements,
            "gather_linear: index {} (position {}) exceeds source length {}",
            source_index,
            position,
            source_elements
        );
        let start = source_index * lane_factor;
        output.extend_from_slice(&data[start..start + lane_factor]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_and_interleaved_lanes_follow_logical_indices() {
        assert_eq!(
            floating_elements(&[1.0_f32, 2.0, 3.0], 1, 3, &[2, 0], 2).unwrap(),
            vec![3.0, 1.0]
        );
        assert_eq!(
            floating_elements(&[1.0_f32, -1.0, 2.0, -2.0], 2, 2, &[1, 0], 2).unwrap(),
            vec![2.0, -2.0, 1.0, -1.0]
        );
    }

    #[test]
    fn malformed_geometry_and_index_are_errors() {
        assert!(floating_elements(&[1.0_f64], 1, 1, &[0], 2).is_err());
        assert!(floating_elements(&[1.0_f64], 1, 2, &[0], 1).is_err());
        assert!(floating_elements(&[1.0_f64], 1, 1, &[1], 1).is_err());
        assert!(floating_elements::<f64>(&[], usize::MAX, 2, &[], 0).is_err());
    }
}

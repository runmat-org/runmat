//! Pure column-major addressing shared by provider execution and recipes.
//! This does not authenticate an allocation or grant access to physical bytes.
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::TransposeInfo;

pub(crate) fn resolve(
    source_shape: &[usize],
    transpose: Option<TransposeInfo>,
    output_shape: &[usize],
    max_rank: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let rank = output_shape.len();
    ensure!(
        rank <= max_rank,
        "fusion broadcast rank exceeds provider limit"
    );
    let mut shape = runmat_value::exact_capacity_vec_with(rank, |_| 0u32)?;
    let mut strides = runmat_value::exact_capacity_vec_with(rank, |_| 0u32)?;
    resolve_into(
        source_shape,
        transpose,
        output_shape,
        max_rank,
        &mut shape,
        &mut strides,
    )?;
    Ok((shape, strides))
}

/// Same canonical indexing calculation with caller-owned exact-rank slots.
/// This performs no allocation and conveys no physical access authority.
pub(crate) fn resolve_into(
    source_shape: &[usize],
    transpose: Option<TransposeInfo>,
    output_shape: &[usize],
    max_rank: usize,
    shape: &mut [u32],
    strides: &mut [u32],
) -> Result<()> {
    let rank = output_shape.len();
    ensure!(
        rank <= max_rank,
        "fusion broadcast rank exceeds provider limit"
    );
    ensure!(
        shape.len() == rank && strides.len() == rank,
        "fusion operand output slot rank mismatch"
    );
    let extent = |axis: usize| {
        if rank >= 2 && source_shape.len() == 1 {
            if axis == 1 {
                source_shape[0]
            } else {
                1
            }
        } else {
            source_shape.get(axis).copied().unwrap_or(1)
        }
    };
    ensure!(
        source_shape.iter().skip(rank).all(|extent| *extent == 1),
        "fusion operand rank exceeds output rank"
    );
    ensure!(
        output_shape
            .iter()
            .enumerate()
            .all(|(axis, output)| extent(axis) == 1 || extent(axis) == *output),
        "fusion operand does not broadcast to output"
    );
    if let Some(info) = transpose {
        ensure!(
            source_shape.len() == 2 && rank >= 2,
            "fusion lazy transpose requires a matrix view"
        );
        ensure!(
            source_shape == [info.base_cols, info.base_rows],
            "fusion transpose metadata mismatch"
        );
    }
    let mut stride = 1usize;
    for (axis, output_stride) in strides.iter_mut().enumerate().take(rank) {
        let input_extent = extent(axis);
        *output_stride = 0;
        if input_extent != 1 {
            let logical_stride = match (transpose, axis) {
                (Some(info), 0) => info.base_rows,
                (Some(_), 1) => 1,
                _ => stride,
            };
            *output_stride = u32::try_from(logical_stride)
                .map_err(|_| anyhow!("fusion input stride exceeds GPU index limit"))?;
        }
        stride = stride
            .checked_mul(input_extent)
            .ok_or_else(|| anyhow!("fusion input stride overflow"))?;
    }
    for (axis, output) in shape.iter_mut().enumerate() {
        *output = u32::try_from(extent(axis))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_owned_slots_preserve_vector_transpose_zero_and_rejection_geometry() {
        for (source, output, expected_shape, expected_strides) in [
            (&[3][..], &[2, 3][..], &[1, 3][..], &[0, 1][..]),
            (&[1, 2, 1][..], &[4, 2][..], &[1, 2][..], &[0, 1][..]),
            (&[0, 1][..], &[0, 4][..], &[0, 1][..], &[1, 0][..]),
            (&[][..], &[][..], &[][..], &[][..]),
        ] {
            let (shape, strides) = resolve(source, None, output, 4).unwrap();
            assert_eq!(shape, expected_shape);
            assert_eq!(strides, expected_strides);
            let mut slots = vec![0; output.len()];
            let mut steps = vec![0; output.len()];
            resolve_into(source, None, output, 4, &mut slots, &mut steps).unwrap();
            assert_eq!((slots, steps), (shape, strides));
        }
        let (shape, strides) = resolve(
            &[3, 2],
            Some(TransposeInfo {
                base_rows: 2,
                base_cols: 3,
            }),
            &[3, 2],
            2,
        )
        .unwrap();
        assert_eq!(shape, [3, 2]);
        assert_eq!(strides, [2, 1]);
        assert!(resolve(&[2], None, &[3, 4], 2).is_err());
        assert!(resolve(&[2, 3], None, &[2, 3], 1).is_err());
        assert!(resolve_into(&[2, 3], None, &[2, 3], 2, &mut [0], &mut [0, 0]).is_err());
    }
}

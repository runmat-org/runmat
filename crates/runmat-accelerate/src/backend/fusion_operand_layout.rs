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
    let mut shape = if rank >= 2 {
        match source_shape.len() {
            0 => vec![1, 1],
            1 => vec![1, source_shape[0]],
            _ => source_shape.to_vec(),
        }
    } else {
        source_shape.to_vec()
    };
    ensure!(
        shape.iter().skip(rank).all(|extent| *extent == 1),
        "fusion operand rank exceeds output rank"
    );
    shape.resize(rank, 1);
    ensure!(
        shape
            .iter()
            .zip(output_shape)
            .all(|(input, output)| *input == 1 || input == output),
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
    let mut strides = vec![0; rank];
    let mut stride = 1usize;
    for axis in 0..rank {
        if shape[axis] != 1 {
            let logical_stride = match (transpose, axis) {
                (Some(info), 0) => info.base_rows,
                (Some(_), 1) => 1,
                _ => stride,
            };
            strides[axis] = u32::try_from(logical_stride)
                .map_err(|_| anyhow!("fusion input stride exceeds GPU index limit"))?;
        }
        stride = stride
            .checked_mul(shape[axis])
            .ok_or_else(|| anyhow!("fusion input stride overflow"))?;
    }
    Ok((
        shape
            .into_iter()
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?,
        strides,
    ))
}

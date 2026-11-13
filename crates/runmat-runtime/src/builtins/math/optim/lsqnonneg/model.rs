//! Original physical model plus one scaled coefficient workspace; x is never redefined publicly.
use super::*;
use runmat_value::{HostNumericBuffer, NumericStorage, SparseTensor};
pub(super) struct Model {
    pub c: SparseTensor,
    pub scaled: SparseTensor,
    pub d: Vec<f64>,
    pub scale: f64,
    pub default_threshold: f64,
}
impl Model {
    pub async fn new(
        c: SparseTensor,
        d: Vec<f64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let values = c.as_f64_slice().expect("NNLS normalized binary64");
        let mut scale = 0.0_f64;
        for chunk in values.chunks(1024) {
            for value in chunk {
                scale = scale.max(value.abs());
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        let scaled = if scale == 0.0 {
            c.clone()
        } else {
            let mut data = runmat_value::exact_capacity_vec(values.len()).map_err(allocation)?;
            for chunk in values.chunks(1024) {
                for &value in chunk {
                    let normalized = value / scale;
                    if value != 0.0 && normalized == 0.0 {
                        return Err(numerical(
                            "coefficient normalization loses a nonzero component",
                        ));
                    }
                    data.push(normalized);
                }
                cooperation.completed(chunk.len() as u64).await?;
            }
            SparseTensor::from_host_numeric_buffers(
                c.rows,
                c.cols,
                c.col_ptrs.clone(),
                c.row_indices.clone(),
                HostNumericBuffer::from_numeric_storage(NumericStorage::F64(data)),
            )
            .map_err(invalid)?
        };
        let mut norm_one = 0.0_f64;
        let values = scaled.as_f64_slice().expect("scaled binary64");
        for col in 0..c.cols {
            let mut sum = 0.0;
            for k in c.col_ptrs[col]..c.col_ptrs[col + 1] {
                sum += values[k].abs();
                if k % 1024 == 1023 {
                    cooperation.completed(1024).await?;
                }
            }
            norm_one = norm_one.max(sum);
            cooperation.completed(1).await?;
        }
        let default_threshold = (10.0 * f64::EPSILON * c.rows.max(c.cols) as f64) * norm_one;
        if !default_threshold.is_finite() {
            return Err(numerical("default tolerance exceeds numerical range"));
        }
        Ok(Self {
            c,
            scaled,
            d,
            scale,
            default_threshold,
        })
    }
}
pub(super) struct Outcome {
    pub x: Vec<f64>,
    pub residual: Vec<f64>,
    pub resnorm: f64,
    pub lambda: Vec<f64>,
    pub flag: i32,
    pub message: &'static str,
}

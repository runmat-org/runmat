use super::*;
#[derive(Clone)]
pub(super) struct EdgeSets {
    pub(super) data: Vec<f64>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) vector_shape: Option<Vec<usize>>,
}

impl EdgeSets {
    pub(super) fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        let shape = tensor.shape().to_vec();
        let data = tensor::tensor_into_values_f64(tensor);
        if data.is_empty() || is_vector_shape(&shape) {
            validate_edges(&data)?;
            return Ok(Self {
                rows: data.len(),
                cols: 1,
                data,
                vector_shape: Some(shape),
            });
        }
        if shape.len() > 2 {
            return Err(invalid_argument(
                "histc: matrix binranges must be two-dimensional",
            ));
        }
        let rows = shape.first().copied().unwrap_or(data.len());
        let cols = shape.get(1).copied().unwrap_or(1);
        for col in 0..cols {
            let start = col * rows;
            validate_edges(&data[start..start + rows])?;
        }
        Ok(Self {
            data,
            rows,
            cols,
            vector_shape: None,
        })
    }

    pub(super) fn edge_slice(&self, slice: usize) -> &[f64] {
        let col = if self.cols == 1 { 0 } else { slice };
        let start = col * self.rows;
        &self.data[start..start + self.rows]
    }

    pub(super) fn output_shape_for_vector_call(&self) -> Vec<usize> {
        self.vector_shape
            .clone()
            .unwrap_or_else(|| vec![self.rows, self.cols])
    }
}

pub(super) fn validate_edges(edges: &[f64]) -> BuiltinResult<()> {
    for &edge in edges {
        if edge.is_nan() {
            return Err(invalid_argument("histc: binranges must not contain NaN"));
        }
    }
    for pair in edges.windows(2) {
        if pair[1] < pair[0] {
            return Err(invalid_argument(
                "histc: binranges must be monotonically nondecreasing",
            ));
        }
    }
    Ok(())
}

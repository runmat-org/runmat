use super::*;
pub async fn evaluate(
    x: Value,
    binranges: Value,
    dim: Option<usize>,
) -> BuiltinResult<HistcResult> {
    let dim = dim.map(validate_dimension_argument).transpose()?;
    ensure_histc_input_extensions(&x)?;
    ensure_histc_input_extensions(&binranges)?;
    let x = value_to_real_tensor(x).await?;
    let edges = EdgeSets::from_tensor(value_to_real_tensor(binranges).await?)?;
    histc_from_tensors(x, edges, dim)
}

#[derive(Debug)]
pub struct HistcResult {
    pub(super) counts: Tensor,
    pub(super) indices: Tensor,
}

pub(super) fn histc_from_tensors(
    x: Tensor,
    edges: EdgeSets,
    dim: Option<usize>,
) -> BuiltinResult<HistcResult> {
    let x_values = tensor::tensor_values_f64(&x);
    let x_shape = canonical_shape(x.shape());
    let vector_call = dim.is_none() && is_vector_shape(&x_shape);
    let dim_index = match dim {
        Some(dim) => dim - 1,
        None => first_nonsingleton_dim(&x_shape),
    };
    let shape = shape_with_dim(x_shape, dim_index);
    let dim_len = shape[dim_index];
    let stride = checked_product(&shape[..dim_index])?;
    let after = checked_product(&shape[dim_index + 1..])?;
    let slice_count = stride
        .checked_mul(after)
        .ok_or_else(|| internal_error("histc: slice count overflow"))?;

    if edges.vector_shape.is_none() && edges.cols != 1 && edges.cols != slice_count {
        return Err(invalid_argument(format!(
            "histc: binranges has {} columns but operation has {slice_count} slices",
            edges.cols
        )));
    }

    let counts_shape = if vector_call {
        edges.output_shape_for_vector_call()
    } else {
        let mut out = shape.clone();
        out[dim_index] = edges.rows;
        trim_trailing_added_dims(out, x.shape().len(), dim_index)
    };
    let mut counts = vec![0.0; checked_product(&counts_shape)?];
    let empty_edges = edges.rows == 0;
    let mut indices = if empty_edges {
        Vec::new()
    } else {
        vec![0.0; x_values.len()]
    };

    let out_stride = if vector_call {
        1
    } else {
        checked_product(&counts_shape[..dim_index])?
    };
    let out_dim_len = edges.rows;

    for high in 0..after {
        for low in 0..stride {
            let slice = low + high * stride;
            let edge_slice = edges.edge_slice(slice);
            for pos in 0..dim_len {
                let input_index = low + pos * stride + high * stride * dim_len;
                if input_index >= x_values.len() {
                    continue;
                }
                if let Some(bin) = classify_bin(x_values[input_index], edge_slice) {
                    if !empty_edges {
                        indices[input_index] = (bin + 1) as f64;
                    }
                    let count_index = if vector_call {
                        bin
                    } else {
                        low + bin * out_stride + high * out_stride * out_dim_len
                    };
                    if let Some(count) = counts.get_mut(count_index) {
                        *count += 1.0;
                    }
                }
            }
        }
    }

    Ok(HistcResult {
        counts: Tensor::new(counts, counts_shape)
            .map_err(|err| internal_error(format!("histc: {err}")))?,
        indices: Tensor::new(
            indices,
            if empty_edges {
                vec![0, 0]
            } else {
                x.shape().to_vec()
            },
        )
        .map_err(|err| internal_error(format!("histc: {err}")))?,
    })
}

pub(super) fn classify_bin(value: f64, edges: &[f64]) -> Option<usize> {
    if value.is_nan() || edges.is_empty() {
        return None;
    }
    let last = *edges.last()?;
    if value == last {
        return Some(edges.len() - 1);
    }
    if value < edges[0] || value > last {
        return None;
    }
    let insertion = edges.partition_point(|edge| *edge <= value);
    if insertion == 0 {
        return None;
    }
    let bin = insertion - 1;
    (bin < edges.len().saturating_sub(1)).then_some(bin)
}

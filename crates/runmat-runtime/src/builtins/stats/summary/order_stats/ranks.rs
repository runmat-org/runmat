use super::*;
use runmat_builtins::TiedRankMode;

fn tiedrank_slice(values: &[OrderedValue], mode: TiedRankMode) -> (Vec<f64>, [f64; 3]) {
    let mut indexed = values
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .collect::<Vec<_>>();
    indexed.sort_by(|(_, a), (_, b)| compare_ordered_values(a, b));
    let mut ranks = vec![f64::NAN; values.len()];
    let mut tieadj = [0.0; 3];
    let mut start = 0usize;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && ordered_values_equal(&indexed[end].1, &indexed[start].1) {
            end += 1;
        }
        let ordinary_rank = (start + 1 + end) as f64 / 2.0;
        let average_rank = if mode == TiedRankMode::Ansari {
            (start..end)
                .map(|index| (index + 1).min(indexed.len() - index) as f64)
                .sum::<f64>()
                / (end - start) as f64
        } else {
            ordinary_rank
        };
        let tie_len = end - start;
        if tie_len > 1 {
            let t = tie_len as f64;
            tieadj[0] += (t * t * t - t) / 2.0;
            if mode == TiedRankMode::Kendall {
                tieadj[1] += t * (t - 1.0) * (t - 2.0);
                tieadj[2] += t * (t - 1.0) * (2.0 * t + 5.0);
            }
        }
        for (original, _) in &indexed[start..end] {
            ranks[*original] = average_rank;
        }
        start = end;
    }
    (ranks, tieadj)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|dim| **dim > 1).count() <= 1
}

pub(super) fn tiedrank_tensor(input: Tensor, mode: TiedRankMode) -> BuiltinResult<(Value, Value)> {
    let input_values = ordered_input_values(&input);
    let shape = tensor::default_shape_for(input.shape(), input_values.len());
    if is_vector_shape(&shape) {
        let (ranks, tieadj) = tiedrank_slice(&input_values, mode);
        let ranks = Tensor::new(ranks, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| order_error("tiedrank", format!("tiedrank: {err}")))?;
        let adjustment = if mode == TiedRankMode::Kendall {
            Tensor::new(tieadj.to_vec(), vec![3, 1])
                .map(tensor::tensor_into_value)
                .map_err(|err| order_error("tiedrank", format!("tiedrank: {err}")))?
        } else {
            Value::Num(tieadj[0])
        };
        return Ok((ranks, adjustment));
    }

    let axis = if shape.len() <= 2 { 0 } else { 1 };
    let rank = shape.len().max(axis + 1);
    let mut padded_shape = shape.clone();
    padded_shape.resize(rank, 1);
    let axis_len = padded_shape[axis];
    let mut tieadj_shape = padded_shape.clone();
    tieadj_shape[axis] = if mode == TiedRankMode::Kendall { 3 } else { 1 };
    let tieadj_len = tensor::element_count(&tieadj_shape);
    let mut ranks = vec![f64::NAN; input_values.len()];
    let mut tieadj = vec![0.0; tieadj_len];
    let pre: usize = padded_shape[..axis].iter().product();
    let post: usize = padded_shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            let mut slice = Vec::with_capacity(axis_len);
            for idx in 0..axis_len {
                let src = prefix + idx * pre + suffix * pre * axis_len;
                slice.push(input_values[src].clone());
            }
            let (slice_ranks, slice_tieadj) = tiedrank_slice(&slice, mode);
            let adjusted_axis_len = tieadj_shape[axis];
            for (component, value) in slice_tieadj.into_iter().take(adjusted_axis_len).enumerate() {
                let tie_dst = prefix + component * pre + suffix * pre * adjusted_axis_len;
                tieadj[tie_dst] = value;
            }
            for (idx, rank_value) in slice_ranks.into_iter().enumerate() {
                let dst = prefix + idx * pre + suffix * pre * axis_len;
                ranks[dst] = rank_value;
            }
        }
    }
    let ranks = Tensor::new(ranks, padded_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| order_error("tiedrank", format!("tiedrank: {err}")))?;
    let tieadj = Tensor::new(tieadj, tieadj_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| order_error("tiedrank", format!("tiedrank: {err}")))?;
    Ok((ranks, tieadj))
}

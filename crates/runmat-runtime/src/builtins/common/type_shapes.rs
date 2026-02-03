use runmat_builtins::Type;

pub fn tensor_shape(ty: &Type) -> Option<&[Option<usize>]> {
    match ty {
        Type::Tensor { shape: Some(shape) } => Some(shape.as_slice()),
        _ => None,
    }
}

pub fn scalar_tensor_shape() -> Vec<Option<usize>> {
    vec![Some(1), Some(1)]
}

pub fn unknown_shape(rank: usize) -> Vec<Option<usize>> {
    vec![None; rank]
}

pub fn element_count_if_known(shape: &[Option<usize>]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, dim| match dim {
        Some(value) => acc.checked_mul(*value),
        None => None,
    })
}

pub fn broadcast_shapes(lhs: &[Option<usize>], rhs: &[Option<usize>]) -> Vec<Option<usize>> {
    let max_rank = lhs.len().max(rhs.len());
    let mut out: Vec<Option<usize>> = Vec::with_capacity(max_rank);
    for i in 0..max_rank {
        let lhs_idx = lhs.len().checked_sub(1 + i);
        let rhs_idx = rhs.len().checked_sub(1 + i);
        let da = lhs_idx
            .and_then(|idx| lhs.get(idx))
            .cloned()
            .unwrap_or(Some(1));
        let db = rhs_idx
            .and_then(|idx| rhs.get(idx))
            .cloned()
            .unwrap_or(Some(1));
        let dim = match (da, db) {
            (Some(a), Some(b)) => {
                if a == b {
                    Some(a)
                } else if a == 1 {
                    Some(b)
                } else if b == 1 {
                    Some(a)
                } else {
                    None
                }
            }
            (Some(a), None) => {
                if a == 1 {
                    None
                } else {
                    Some(a)
                }
            }
            (None, Some(b)) => {
                if b == 1 {
                    None
                } else {
                    Some(b)
                }
            }
            (None, None) => None,
        };
        out.push(dim);
    }
    out.reverse();
    out
}

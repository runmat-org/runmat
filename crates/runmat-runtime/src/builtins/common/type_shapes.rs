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

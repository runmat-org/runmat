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

pub fn array_shape(ty: &Type) -> Option<&[Option<usize>]> {
    match ty {
        Type::Tensor { shape: Some(shape) } => Some(shape.as_slice()),
        Type::Logical { shape: Some(shape) } => Some(shape.as_slice()),
        _ => None,
    }
}

pub fn element_count_if_known(shape: &[Option<usize>]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, dim| match dim {
        Some(value) => acc.checked_mul(*value),
        None => None,
    })
}

pub fn squeeze_shape_options(shape: &[Option<usize>]) -> Vec<Option<usize>> {
    if shape.len() <= 2 {
        return shape.to_vec();
    }
    let mut squeezed: Vec<Option<usize>> = shape
        .iter()
        .copied()
        .filter(|dim| *dim != Some(1))
        .collect();
    if squeezed.is_empty() {
        squeezed = vec![Some(1), Some(1)];
    } else if squeezed.len() == 1 {
        squeezed.push(Some(1));
    }
    squeezed
}

pub fn reshape_rank_from_args(args: &[Type]) -> Option<usize> {
    if args.len() < 2 {
        return None;
    }
    if args.len() > 2 {
        return Some(args.len() - 1);
    }
    match &args[1] {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape)
        }
        Type::Num | Type::Int | Type::Bool => Some(1),
        _ => None,
    }
}

pub fn permute_order_len(ty: &Type) -> Option<usize> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape)
        }
        Type::Num | Type::Int | Type::Bool => Some(1),
        _ => None,
    }
}

pub fn repmat_reps_len(args: &[Type]) -> Option<usize> {
    if args.len() < 2 {
        return None;
    }
    if args.len() > 2 {
        return Some(args.len() - 1);
    }
    match &args[1] {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape)
        }
        Type::Num | Type::Int | Type::Bool => Some(1),
        _ => None,
    }
}

pub fn concat_shape(
    shapes: &[Vec<Option<usize>>],
    dim_1based: usize,
) -> Option<Vec<Option<usize>>> {
    if shapes.is_empty() {
        return None;
    }
    if dim_1based == 0 {
        return None;
    }
    let rank = shapes
        .iter()
        .map(|shape| shape.len())
        .max()?
        .max(dim_1based);
    let mut padded = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let mut current = shape.clone();
        while current.len() < rank {
            current.push(Some(1));
        }
        padded.push(current);
    }

    let mut output = vec![None; rank];
    let dim_zero = dim_1based - 1;
    for axis in 0..rank {
        if axis == dim_zero {
            let mut total: Option<usize> = Some(0);
            for shape in &padded {
                match (total, shape[axis]) {
                    (Some(acc), Some(value)) => total = acc.checked_add(value),
                    _ => {
                        total = None;
                        break;
                    }
                }
            }
            output[axis] = total;
        } else {
            let mut shared: Option<usize> = None;
            let mut mismatch = false;
            for shape in &padded {
                match (shared, shape[axis]) {
                    (None, value) => shared = value,
                    (Some(current), Some(value)) if current == value => {}
                    (Some(_), Some(_)) => {
                        mismatch = true;
                        break;
                    }
                    _ => {
                        shared = None;
                        break;
                    }
                }
            }
            output[axis] = if mismatch { None } else { shared };
        }
    }

    let min_len = dim_1based.max(2).min(output.len());
    while output.len() > min_len && matches!(output.last(), Some(Some(1))) {
        output.pop();
    }
    Some(output)
}

pub fn concat_input_shape(ty: &Type) -> Option<Vec<Option<usize>>> {
    match ty {
        Type::Tensor { shape: Some(shape) } => Some(shape.clone()),
        Type::Logical { shape: Some(shape) } => Some(shape.clone()),
        Type::Num | Type::Int | Type::Bool => Some(scalar_tensor_shape()),
        _ => None,
    }
}

pub fn repmat_output_shape(
    input_shape: &[Option<usize>],
    reps_len: usize,
) -> Option<Vec<Option<usize>>> {
    let input_rank = input_shape.len();
    let rank = if reps_len == 1 {
        if input_rank == 0 {
            return None;
        }
        input_rank.max(2)
    } else {
        input_rank.max(reps_len)
    };

    let mut output = Vec::with_capacity(rank);
    for axis in 0..rank {
        if axis < input_rank && input_shape[axis] == Some(0) {
            output.push(Some(0));
        } else {
            output.push(None);
        }
    }
    Some(output)
}

pub fn cell_element_type(inputs: &[Type]) -> Option<Box<Type>> {
    let mut element: Option<Type> = None;
    for ty in inputs {
        let Type::Cell { element_type, .. } = ty else {
            return None;
        };
        match (&element, element_type.as_deref()) {
            (None, Some(current)) => element = Some(current.clone()),
            (Some(existing), Some(current)) if existing == current => {}
            (Some(_), Some(_)) => return None,
            _ => {}
        }
    }
    element.map(Box::new)
}

fn reduction_shape_from_args(args: &[Type]) -> Option<Vec<Option<usize>>> {
    let input = args.first()?;
    let shape = match input {
        Type::Tensor { shape } => shape.clone(),
        Type::Logical { shape } => shape.clone(),
        _ => None,
    };
    let shape = shape?;
    if args.len() == 1 {
        Some(reduce_first_nonsingleton(&shape))
    } else {
        Some(unknown_shape(shape.len()))
    }
}

pub fn reduce_dim(shape: &[Option<usize>], dim_1based: usize) -> Vec<Option<usize>> {
    if dim_1based == 0 {
        return shape.to_vec();
    }
    let idx = dim_1based.saturating_sub(1);
    if idx >= shape.len() {
        return shape.to_vec();
    }
    let mut out = shape.to_vec();
    out[idx] = Some(1);
    out
}

pub fn reduce_first_nonsingleton(shape: &[Option<usize>]) -> Vec<Option<usize>> {
    if shape.is_empty() {
        return scalar_tensor_shape();
    }
    let mut dim = 0usize;
    for (i, entry) in shape.iter().enumerate() {
        if let Some(value) = entry {
            if *value != 1 {
                dim = i;
                break;
            }
        } else {
            dim = 0;
            break;
        }
    }
    let mut out = shape.to_vec();
    if let Some(entry) = out.get_mut(dim) {
        *entry = Some(1);
    }
    out
}

pub fn logical_like(input: &Type) -> Type {
    match input {
        Type::Tensor { shape: Some(shape) } => Type::Logical {
            shape: Some(shape.clone()),
        },
        Type::Tensor { shape: None } => Type::logical(),
        Type::Logical { shape } => Type::Logical {
            shape: shape.clone(),
        },
        Type::Unknown => Type::logical(),
        _ => Type::Bool,
    }
}

pub fn logical_result_for_binary(lhs: &Type, rhs: &Type) -> Type {
    let lhs_shape = match lhs {
        Type::Tensor { shape: Some(shape) } => Some(shape.clone()),
        Type::Logical { shape: Some(shape) } => Some(shape.clone()),
        _ => None,
    };
    let rhs_shape = match rhs {
        Type::Tensor { shape: Some(shape) } => Some(shape.clone()),
        Type::Logical { shape: Some(shape) } => Some(shape.clone()),
        _ => None,
    };
    if let (Some(a), Some(b)) = (&lhs_shape, &rhs_shape) {
        if a == b {
            return Type::Logical {
                shape: Some(a.clone()),
            };
        }
    }
    if let Some(shape) = lhs_shape {
        return Type::Logical { shape: Some(shape) };
    }
    if let Some(shape) = rhs_shape {
        return Type::Logical { shape: Some(shape) };
    }
    if matches!(lhs, Type::Tensor { .. } | Type::Logical { .. })
        || matches!(rhs, Type::Tensor { .. } | Type::Logical { .. })
    {
        Type::logical()
    } else if matches!(lhs, Type::Unknown) || matches!(rhs, Type::Unknown) {
        Type::Unknown
    } else {
        Type::Bool
    }
}

pub fn logical_binary_type(args: &[Type]) -> Type {
    if args.len() >= 2 {
        logical_result_for_binary(&args[0], &args[1])
    } else if let Some(first) = args.first() {
        logical_like(first)
    } else {
        Type::Unknown
    }
}

pub fn logical_unary_type(args: &[Type]) -> Type {
    args.first().map(logical_like).unwrap_or(Type::logical())
}

pub fn bool_scalar_type(_args: &[Type]) -> Type {
    Type::Bool
}

pub fn reduce_numeric_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(_) } | Type::Logical { shape: Some(_) } => Type::Tensor {
            shape: reduction_shape_from_args(args),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Bool | Type::Num | Type::Int => Type::Num,
        _ => Type::Unknown,
    }
}

pub fn reduce_logical_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(_) } | Type::Logical { shape: Some(_) } => Type::Logical {
            shape: reduction_shape_from_args(args),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::logical(),
        Type::Bool | Type::Num | Type::Int => Type::Bool,
        _ => Type::Unknown,
    }
}

pub fn cumulative_numeric_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(shape) } => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Type::Logical { shape: Some(shape) } => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Bool | Type::Num | Type::Int => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn diff_numeric_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    let only_input = args.len() <= 1;
    match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if only_input {
                Type::Tensor {
                    shape: Some(unknown_shape(shape.len())),
                }
            } else {
                Type::tensor()
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Bool | Type::Num | Type::Int => {
            if only_input {
                Type::tensor()
            } else {
                Type::Unknown
            }
        }
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn count_nonzero_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    if args.len() <= 1 {
        match input {
            Type::Tensor { .. } | Type::Logical { .. } | Type::Bool | Type::Num | Type::Int => {
                Type::Num
            }
            Type::Unknown => Type::Unknown,
            _ => Type::Unknown,
        }
    } else {
        reduce_numeric_type(args)
    }
}

pub fn min_max_type(args: &[Type]) -> Type {
    if args.len() <= 1 {
        return reduce_numeric_type(args);
    }
    let mut has_array = false;
    let mut has_unknown = false;
    for arg in args {
        match arg {
            Type::Tensor { .. } | Type::Logical { .. } => has_array = true,
            Type::Unknown => has_unknown = true,
            _ => {}
        }
    }
    if has_array {
        Type::tensor()
    } else if has_unknown {
        Type::Unknown
    } else {
        Type::Num
    }
}

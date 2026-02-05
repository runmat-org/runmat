use runmat_builtins::{ResolveContext, Type};

use runmat_builtins::shape_rules::element_count_if_known;

pub fn conv_type(args: &[Type], _context: &ResolveContext) -> Type {
    conv_like_type(args)
}

pub fn conv2_type(args: &[Type], _context: &ResolveContext) -> Type {
    let lhs = args.get(0);
    let rhs = args.get(1);
    match (lhs, rhs) {
        (Some(left), Some(right)) => conv2_binary_type(left, right, args.len() == 2),
        _ => Type::Unknown,
    }
}

pub fn deconv_type(args: &[Type], _context: &ResolveContext) -> Type {
    let numerator = args.get(0);
    let denominator = args.get(1);
    match (numerator, denominator) {
        (Some(num), Some(den)) => deconv_binary_type(num, den),
        _ => Type::Unknown,
    }
}

pub fn filter_type(args: &[Type], _context: &ResolveContext) -> Type {
    let signal = args.get(2);
    match signal {
        Some(ty) => numeric_like(ty),
        None => Type::Unknown,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrientationHint {
    Row,
    Column,
    Scalar,
    General,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Row,
    Column,
}

fn conv_like_type(args: &[Type]) -> Type {
    let lhs = args.get(0);
    let rhs = args.get(1);
    match (lhs, rhs) {
        (Some(left), Some(right)) => conv_binary_type(left, right, args.len() == 2),
        _ => Type::Unknown,
    }
}

fn conv_binary_type(lhs: &Type, rhs: &Type, is_default_mode: bool) -> Type {
    if is_numeric_scalar(lhs) && is_numeric_scalar(rhs) {
        return Type::Num;
    }

    let lhs_len = vector_len(lhs);
    let rhs_len = vector_len(rhs);
    let lhs_hint = orientation_hint(lhs);
    let rhs_hint = orientation_hint(rhs);
    let orientation = output_orientation(lhs_hint, rhs_hint);

    let out_len = if is_default_mode {
        match (lhs_len, rhs_len) {
            (Some(a), Some(b)) => Some(a + b - 1),
            _ => None,
        }
    } else {
        None
    };
    vector_output_type(out_len, orientation)
}

fn conv2_binary_type(lhs: &Type, rhs: &Type, is_default_mode: bool) -> Type {
    let (lhs_rows, lhs_cols) = matrix_dims(lhs);
    let (rhs_rows, rhs_cols) = matrix_dims(rhs);
    let (rows, cols) = if is_default_mode {
        match (lhs_rows, lhs_cols, rhs_rows, rhs_cols) {
            (Some(a), Some(b), Some(c), Some(d)) => (Some(a + c - 1), Some(b + d - 1)),
            _ => (None, None),
        }
    } else {
        (None, None)
    };
    Type::Tensor {
        shape: Some(vec![rows, cols]),
    }
}

fn deconv_binary_type(numerator: &Type, denominator: &Type) -> Type {
    if is_numeric_scalar(numerator) && is_numeric_scalar(denominator) {
        return Type::Num;
    }
    let num_len = vector_len(numerator);
    let den_len = vector_len(denominator);
    let hint = orientation_hint(numerator);
    let out_len = match (num_len, den_len) {
        (Some(a), Some(b)) if a >= b => Some(a - b + 1),
        (Some(_), Some(_)) => Some(0),
        _ => None,
    };
    vector_output_type(out_len, orientation_from_hint(hint))
}

fn vector_output_type(len: Option<usize>, orientation: Orientation) -> Type {
    let shape = match (orientation, len) {
        (Orientation::Row, Some(0)) => vec![Some(1), Some(0)],
        (Orientation::Column, Some(0)) => vec![Some(0), Some(1)],
        (Orientation::Row, Some(n)) => vec![Some(1), Some(n)],
        (Orientation::Column, Some(n)) => vec![Some(n), Some(1)],
        (Orientation::Row, None) => vec![Some(1), None],
        (Orientation::Column, None) => vec![None, Some(1)],
    };
    Type::Tensor { shape: Some(shape) }
}

fn output_orientation(lhs: OrientationHint, rhs: OrientationHint) -> Orientation {
    match lhs {
        OrientationHint::Row => Orientation::Row,
        OrientationHint::Column => Orientation::Column,
        OrientationHint::General => Orientation::Column,
        OrientationHint::Scalar | OrientationHint::Empty => orientation_from_hint(rhs),
    }
}

fn orientation_from_hint(hint: OrientationHint) -> Orientation {
    match hint {
        OrientationHint::Column | OrientationHint::General => Orientation::Column,
        OrientationHint::Row => Orientation::Row,
        OrientationHint::Scalar | OrientationHint::Empty => Orientation::Row,
    }
}

fn orientation_hint(ty: &Type) -> OrientationHint {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            let len = element_count_if_known(shape);
            let (rows, cols) = shape_rows_cols(shape);
            classify_orientation(rows, cols, len)
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => OrientationHint::General,
        Type::Num | Type::Int | Type::Bool => OrientationHint::Scalar,
        _ => OrientationHint::General,
    }
}

fn classify_orientation(
    rows: Option<usize>,
    cols: Option<usize>,
    len: Option<usize>,
) -> OrientationHint {
    if len == Some(0) {
        OrientationHint::Empty
    } else if rows == Some(1) && cols.unwrap_or(0) > 1 {
        OrientationHint::Row
    } else if cols == Some(1) && rows.unwrap_or(0) > 1 {
        OrientationHint::Column
    } else if rows == Some(1) && cols == Some(1) {
        OrientationHint::Scalar
    } else {
        OrientationHint::General
    }
}

fn shape_rows_cols(shape: &[Option<usize>]) -> (Option<usize>, Option<usize>) {
    match shape.len() {
        0 => (Some(1), Some(1)),
        1 => (Some(1), shape.get(0).copied().flatten()),
        _ => (
            shape.get(0).copied().flatten(),
            shape.get(1).copied().flatten(),
        ),
    }
}

fn matrix_dims(ty: &Type) -> (Option<usize>, Option<usize>) {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if shape.len() > 2 && shape.iter().skip(2).any(|dim| dim != &Some(1)) {
                return (None, None);
            }
            match shape.len() {
                0 => (Some(1), Some(1)),
                1 => (shape.get(0).copied().flatten(), Some(1)),
                _ => (
                    shape.get(0).copied().flatten(),
                    shape.get(1).copied().flatten(),
                ),
            }
        }
        Type::Num | Type::Int | Type::Bool => (Some(1), Some(1)),
        Type::Tensor { shape: None } | Type::Logical { shape: None } => (None, None),
        _ => (None, None),
    }
}

fn vector_len(ty: &Type) -> Option<usize> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape)
        }
        Type::Num | Type::Int | Type::Bool => Some(1),
        _ => None,
    }
}

fn numeric_like(input: &Type) -> Type {
    match input {
        Type::Tensor { shape: Some(shape) } => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Type::Logical { shape: Some(shape) } => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn is_numeric_scalar(ty: &Type) -> bool {
    match ty {
        Type::Num | Type::Int | Type::Bool => true,
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape.as_slice()) == Some(1)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_full_uses_length_sum() {
        let out = conv_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn filter_preserves_signal_shape() {
        let out = filter_type(
            &[
                Type::Num,
                Type::Num,
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(2)])
            }
        );
    }
}

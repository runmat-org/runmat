use runmat_builtins::Type;

use runmat_builtins::shape_rules::element_count_if_known;

pub fn polyder_type(args: &[Type]) -> Type {
    if args.is_empty() {
        return Type::tensor();
    }
    if args.iter().all(is_scalar_type) {
        return Type::Num;
    }
    match args.first() {
        Some(Type::Tensor { shape: Some(shape) }) | Some(Type::Logical { shape: Some(shape) }) => {
            match element_count_if_known(shape) {
                Some(1) => Type::Num,
                _ => Type::tensor(),
            }
        }
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Tensor { shape: None } | Type::Logical { shape: None }) => Type::tensor(),
        _ => Type::tensor(),
    }
}

pub fn polyfit_type(_args: &[Type]) -> Type {
    Type::tensor()
}

pub fn polyint_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::tensor();
    };
    match input {
        Type::Num | Type::Int | Type::Bool => Type::tensor_with_shape(vec![1, 2]),
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            match element_count_if_known(shape) {
                Some(1) => Type::tensor_with_shape(vec![1, 2]),
                _ => Type::tensor(),
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Unknown => Type::tensor(),
        _ => Type::tensor(),
    }
}

pub fn polyval_type(args: &[Type]) -> Type {
    let points = args.get(1);
    match points {
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Tensor { shape: Some(shape) }) | Some(Type::Logical { shape: Some(shape) }) => {
            match element_count_if_known(shape) {
                Some(1) => Type::Num,
                _ => Type::Tensor {
                    shape: Some(shape.clone()),
                },
            }
        }
        Some(Type::Tensor { shape: None } | Type::Logical { shape: None }) => Type::tensor(),
        Some(Type::Unknown) | None => Type::Union(vec![Type::Num, Type::tensor()]),
        _ => Type::tensor(),
    }
}

pub fn roots_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::tensor();
    };
    match input {
        Type::Num | Type::Int | Type::Bool => Type::tensor_with_shape(vec![0, 1]),
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            match element_count_if_known(shape) {
                Some(0) | Some(1) => Type::tensor_with_shape(vec![0, 1]),
                _ => Type::tensor(),
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        _ => Type::tensor(),
    }
}

fn is_scalar_type(ty: &Type) -> bool {
    match ty {
        Type::Num | Type::Int | Type::Bool => true,
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            matches!(element_count_if_known(shape), Some(1))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyder_type_reports_num_for_scalar() {
        assert_eq!(polyder_type(&[Type::Num]), Type::Num);
    }

    #[test]
    fn polyder_type_reports_num_for_scalar_tensor() {
        assert_eq!(
            polyder_type(&[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)])
            }]),
            Type::Num
        );
    }

    #[test]
    fn polyder_type_reports_num_for_scalar_product() {
        assert_eq!(polyder_type(&[Type::Num, Type::Int]), Type::Num);
    }

    #[test]
    fn polyfit_type_reports_tensor() {
        assert_eq!(polyfit_type(&[]), Type::tensor());
    }

    #[test]
    fn polyint_type_reports_row_for_scalar() {
        assert_eq!(
            polyint_type(&[Type::Num]),
            Type::tensor_with_shape(vec![1, 2])
        );
    }

    #[test]
    fn polyval_type_reports_num_for_scalar_points() {
        assert_eq!(polyval_type(&[Type::tensor(), Type::Num]), Type::Num);
    }

    #[test]
    fn polyval_type_preserves_tensor_shape() {
        assert_eq!(
            polyval_type(&[
                Type::tensor(),
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)])
                }
            ]),
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn roots_type_reports_empty_column_for_scalar() {
        assert_eq!(
            roots_type(&[Type::Num]),
            Type::tensor_with_shape(vec![0, 1])
        );
    }
}

use runmat_builtins::Type;

use runmat_builtins::shape_rules::element_count_if_known;

pub fn pause_type(args: &[Type]) -> Type {
    if args.is_empty() {
        return Type::tensor_with_shape(vec![0, 0]);
    }
    if args.len() > 1 {
        return Type::Unknown;
    }

    match &args[0] {
        Type::String => Type::Union(vec![Type::String, Type::tensor_with_shape(vec![0, 0])]),
        Type::Num | Type::Int | Type::Bool => Type::tensor_with_shape(vec![0, 0]),
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            match element_count_if_known(shape) {
                Some(0) | Some(1) => Type::tensor_with_shape(vec![0, 0]),
                _ => Type::tensor(),
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Unknown => Type::Union(vec![Type::String, Type::tensor_with_shape(vec![0, 0])]),
        _ => Type::Unknown,
    }
}

pub fn tic_type(_args: &[Type]) -> Type {
    Type::Num
}

pub fn timeit_type(_args: &[Type]) -> Type {
    Type::Num
}

pub fn toc_type(_args: &[Type]) -> Type {
    Type::Num
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_type_reports_empty_tensor_with_no_args() {
        assert_eq!(pause_type(&[]), Type::tensor_with_shape(vec![0, 0]));
    }

    #[test]
    fn pause_type_reports_empty_tensor_for_numeric_arg() {
        assert_eq!(
            pause_type(&[Type::Num]),
            Type::tensor_with_shape(vec![0, 0])
        );
    }

    #[test]
    fn pause_type_reports_union_for_string_arg() {
        assert_eq!(
            pause_type(&[Type::String]),
            Type::Union(vec![Type::String, Type::tensor_with_shape(vec![0, 0])])
        );
    }

    #[test]
    fn pause_type_reports_empty_tensor_for_scalar_tensor() {
        assert_eq!(
            pause_type(&[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)])
            }]),
            Type::tensor_with_shape(vec![0, 0])
        );
    }

    #[test]
    fn tic_type_reports_num() {
        assert_eq!(tic_type(&[]), Type::Num);
    }

    #[test]
    fn timeit_type_reports_num() {
        assert_eq!(timeit_type(&[]), Type::Num);
    }

    #[test]
    fn toc_type_reports_num() {
        assert_eq!(toc_type(&[]), Type::Num);
    }
}

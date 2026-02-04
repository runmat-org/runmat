use runmat_builtins::Type;

use runmat_builtins::shape_rules::element_count_if_known;

pub fn filter2_type(args: &[Type]) -> Type {
    if args.len() != 2 {
        return Type::tensor();
    }
    image_shape_output(&args[1]).unwrap_or_else(Type::tensor)
}

pub fn fspecial_type(_args: &[Type]) -> Type {
    Type::tensor()
}

pub fn imfilter_type(args: &[Type]) -> Type {
    if args.len() != 2 {
        return Type::tensor();
    }
    image_shape_output(&args[0]).unwrap_or_else(Type::tensor)
}

fn image_shape_output(image: &Type) -> Option<Type> {
    match image {
        Type::Num | Type::Int | Type::Bool => Some(Type::Num),
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            match element_count_if_known(shape) {
                Some(1) => Some(Type::Num),
                _ => Some(Type::Tensor {
                    shape: Some(shape.clone()),
                }),
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Some(Type::tensor()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter2_type_reports_tensor_without_args() {
        assert_eq!(filter2_type(&[]), Type::tensor());
    }

    #[test]
    fn filter2_type_preserves_image_shape_when_defaulted() {
        assert_eq!(
            filter2_type(&[
                Type::tensor(),
                Type::Tensor {
                    shape: Some(vec![Some(4), Some(5)])
                }
            ]),
            Type::Tensor {
                shape: Some(vec![Some(4), Some(5)])
            }
        );
    }

    #[test]
    fn fspecial_type_reports_tensor() {
        assert_eq!(fspecial_type(&[]), Type::tensor());
    }

    #[test]
    fn imfilter_type_preserves_image_shape_when_defaulted() {
        assert_eq!(
            imfilter_type(&[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)])
                },
                Type::tensor()
            ]),
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }
}

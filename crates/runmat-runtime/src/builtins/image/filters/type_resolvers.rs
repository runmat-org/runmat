use runmat_builtins::shape_rules::element_count_if_known;
use runmat_builtins::{LiteralValue, ResolveContext, Type};

pub fn filter2_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.len() != 2 {
        return Type::tensor();
    }
    image_shape_output(&args[1]).unwrap_or_else(Type::tensor)
}

pub fn fspecial_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn imfilter_type(args: &[Type], context: &ResolveContext) -> Type {
    if args.len() < 2 {
        return Type::tensor();
    }
    if args.len() > 2 {
        let options = context.literal_args.get(2..).unwrap_or(&[]);
        if options.len() != args.len() - 2
            || options.iter().any(|option| match option {
                LiteralValue::String(value) => {
                    matches!(value.trim().to_ascii_lowercase().as_str(), "full" | "valid")
                }
                LiteralValue::Number(_) | LiteralValue::Bool(_) => false,
                LiteralValue::Vector(_) | LiteralValue::Unknown => true,
            })
        {
            return Type::tensor();
        }
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
        assert_eq!(
            filter2_type(&[], &ResolveContext::new(Vec::new())),
            Type::tensor()
        );
    }

    #[test]
    fn filter2_type_preserves_image_shape_when_defaulted() {
        assert_eq!(
            filter2_type(
                &[
                    Type::tensor(),
                    Type::Tensor {
                        shape: Some(vec![Some(4), Some(5)])
                    }
                ],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Tensor {
                shape: Some(vec![Some(4), Some(5)])
            }
        );
    }

    #[test]
    fn fspecial_type_reports_tensor() {
        assert_eq!(
            fspecial_type(&[], &ResolveContext::new(Vec::new())),
            Type::tensor()
        );
    }

    #[test]
    fn imfilter_type_preserves_image_shape_when_defaulted() {
        assert_eq!(
            imfilter_type(
                &[
                    Type::Tensor {
                        shape: Some(vec![Some(2), Some(3)])
                    },
                    Type::tensor()
                ],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn imfilter_type_preserves_only_provably_same_shape_option_forms() {
        let args = [
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            },
            Type::tensor(),
            Type::tensor(),
        ];
        let same = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("replicate".to_string()),
        ]);
        assert_eq!(
            imfilter_type(&args, &same),
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );

        for option in [
            LiteralValue::String("full".to_string()),
            LiteralValue::String("valid".to_string()),
            LiteralValue::Unknown,
        ] {
            let context =
                ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Unknown, option]);
            assert_eq!(imfilter_type(&args, &context), Type::tensor());
        }
    }
}

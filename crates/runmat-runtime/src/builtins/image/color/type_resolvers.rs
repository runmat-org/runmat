use runmat_builtins::{ResolveContext, Type};

use super::common;

pub fn same_shape_type(args: &[Type], _context: &ResolveContext) -> Type {
    common::same_shape_type(args)
}

pub fn rgb2gray_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape: Some(shape) }) if shape.len() == 3 && shape[2] == Some(3) => {
            Type::Tensor {
                shape: Some(vec![shape[0], shape[1]]),
            }
        }
        Some(Type::Tensor { shape: Some(shape) }) if shape.len() == 2 && shape[1] == Some(3) => {
            Type::Tensor {
                shape: Some(shape.clone()),
            }
        }
        Some(Type::Tensor { .. }) => Type::tensor(),
        _ => Type::tensor(),
    }
}

pub fn gray2rgb_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape: Some(shape) }) if shape.len() == 2 => Type::Tensor {
            shape: Some(vec![shape[0], shape[1], Some(3)]),
        },
        Some(Type::Logical { shape: Some(shape) }) if shape.len() == 2 => Type::Tensor {
            shape: Some(vec![shape[0], shape[1], Some(3)]),
        },
        _ => Type::tensor(),
    }
}

pub fn ind2rgb_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape: Some(shape) }) | Some(Type::Logical { shape: Some(shape) })
            if shape.len() == 2 =>
        {
            let mut out = shape.clone();
            out.push(Some(3));
            Type::Tensor { shape: Some(out) }
        }
        Some(Type::Num) | Some(Type::Int) | Some(Type::Bool) => Type::Tensor {
            shape: Some(vec![Some(1), Some(1), Some(3)]),
        },
        _ => Type::tensor(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ind2rgb_type_appends_color_plane_only_to_matrix_input() {
        let context = ResolveContext::new(Vec::new());
        assert_eq!(
            ind2rgb_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(2), Some(4)])
                }],
                &context,
            ),
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4), Some(3)])
            }
        );
        assert_eq!(
            ind2rgb_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(2), Some(4), Some(1)])
                }],
                &context,
            ),
            Type::tensor()
        );
    }

    #[test]
    fn rgb2gray_type_only_removes_a_known_three_channel_dimension() {
        let context = ResolveContext::new(Vec::new());
        assert_eq!(
            rgb2gray_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(2), Some(4), Some(3)])
                }],
                &context,
            ),
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4)])
            }
        );
        assert_eq!(
            rgb2gray_type(
                &[Type::Tensor {
                    shape: Some(vec![Some(2), Some(4), Some(4)])
                }],
                &context,
            ),
            Type::tensor()
        );
    }
}

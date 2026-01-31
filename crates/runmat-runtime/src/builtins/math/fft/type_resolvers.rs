use runmat_builtins::Type;

use crate::builtins::common::type_shapes::{element_count_if_known, unknown_shape};

pub fn fft_type(args: &[Type]) -> Type {
    fft_like_type(args, 1)
}

pub fn ifft_type(args: &[Type]) -> Type {
    fft_like_type(args, 1)
}

pub fn fft2_type(args: &[Type]) -> Type {
    fft_like_type(args, 2)
}

pub fn ifft2_type(args: &[Type]) -> Type {
    fft_like_type(args, 2)
}

pub fn fftshift_type(args: &[Type]) -> Type {
    preserve_input_type(args)
}

pub fn ifftshift_type(args: &[Type]) -> Type {
    preserve_input_type(args)
}

fn fft_like_type(args: &[Type], min_rank: usize) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            let rank = shape.len().max(min_rank);
            if args.len() > 1 {
                return Type::Tensor {
                    shape: Some(unknown_shape(rank)),
                };
            }
            let mut out_shape = shape.clone();
            while out_shape.len() < min_rank {
                out_shape.push(Some(1));
            }
            numeric_tensor_from_shape(out_shape)
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn preserve_input_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { shape: Some(shape) } => Type::Tensor {
            shape: Some(shape.clone()),
        },
        Type::Logical { shape: Some(shape) } => Type::Logical {
            shape: Some(shape.clone()),
        },
        Type::Tensor { shape: None } => Type::tensor(),
        Type::Logical { shape: None } => Type::logical(),
        Type::Num | Type::Int | Type::Bool => input.clone(),
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn numeric_tensor_from_shape(shape: Vec<Option<usize>>) -> Type {
    if element_count_if_known(&shape) == Some(1) {
        Type::Num
    } else {
        Type::Tensor { shape: Some(shape) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_like_preserves_shape_without_args() {
        let out = fft_type(&[Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        }]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn fft2_pads_rank() {
        let out = fft2_type(&[Type::Tensor {
            shape: Some(vec![Some(4)]),
        }]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(1)])
            }
        );
    }
}

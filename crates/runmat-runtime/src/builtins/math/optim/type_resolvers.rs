use runmat_builtins::{ResolveContext, Type};

#[cfg(test)]
pub(crate) fn numerical_integral_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Num
}

pub fn nonlinear_solve_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.get(1) {
        Some(Type::Tensor { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        _ => Type::tensor(),
    }
}

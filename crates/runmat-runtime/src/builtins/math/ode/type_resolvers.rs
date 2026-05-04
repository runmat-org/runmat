use runmat_builtins::{ResolveContext, Type};

pub fn ode_solution_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.get(2) {
        Some(Type::Tensor { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::tensor_with_shape(vec![1, 1]),
        _ => Type::tensor(),
    }
}

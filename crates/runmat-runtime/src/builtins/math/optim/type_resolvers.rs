use runmat_builtins::{ResolveContext, Type};

pub fn scalar_root_type(_args: &[Type], _context: &ResolveContext) -> Type {
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

pub fn optim_options_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Struct { known_fields: None }
}

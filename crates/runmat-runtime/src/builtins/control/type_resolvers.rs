use runmat_builtins::{ResolveContext, Type};

use crate::builtins::math::type_resolvers::{numeric_binary_type, numeric_unary_type};

pub fn db_type(args: &[Type], context: &ResolveContext) -> Type {
    match args {
        [input] => numeric_unary_type(std::slice::from_ref(input), context),
        [input, Type::String] => numeric_unary_type(std::slice::from_ref(input), context),
        [input, reference] => numeric_binary_type(&[input.clone(), reference.clone()], context),
        _ => Type::Unknown,
    }
}

pub fn step_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn tf_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

pub fn impulse_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

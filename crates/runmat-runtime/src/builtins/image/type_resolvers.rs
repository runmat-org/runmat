use runmat_builtins::{ResolveContext, Type};

use crate::builtins::array::type_resolvers::column_vector_type;

pub fn imread_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn imwrite_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Void
}

pub fn imhist_type(_args: &[Type], _context: &ResolveContext) -> Type {
    column_vector_type()
}

use runmat_builtins::{ResolveContext, Type};

pub fn tf_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

use runmat_builtins::{ResolveContext, Type};

pub fn warning_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

pub fn error_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

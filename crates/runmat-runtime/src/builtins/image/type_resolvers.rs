use runmat_builtins::{ResolveContext, Type};

pub fn imwrite_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Void
}

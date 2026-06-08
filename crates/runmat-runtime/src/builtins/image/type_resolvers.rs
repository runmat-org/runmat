use runmat_builtins::{ResolveContext, Type};

pub fn imread_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

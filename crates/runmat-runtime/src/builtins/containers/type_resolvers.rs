use runmat_builtins::{ResolveContext, Type};

pub fn map_handle_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

pub fn map_cell_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::cell()
}

pub fn map_is_key_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::logical()
}

pub fn map_unknown_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

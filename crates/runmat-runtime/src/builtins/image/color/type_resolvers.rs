use runmat_builtins::{ResolveContext, Type};

use super::common;

pub fn same_shape_type(args: &[Type], _context: &ResolveContext) -> Type {
    common::same_shape_type(args)
}

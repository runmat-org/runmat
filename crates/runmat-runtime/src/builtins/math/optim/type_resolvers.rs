use runmat_builtins::{ResolveContext, Type};

#[cfg(test)]
pub(crate) fn numerical_integral_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Num
}

//! Transitional resolver for the timing identity not yet catalog-owned.

use runmat_builtins::{ResolveContext, Type};

pub fn timeit_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Num
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeit_type_reports_num() {
        assert_eq!(
            timeit_type(&[], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}

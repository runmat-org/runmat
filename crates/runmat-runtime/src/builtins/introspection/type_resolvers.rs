use runmat_builtins::{ResolveContext, Type};

pub fn ismethod_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

pub fn metaclass_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ismethod_type_reports_bool() {
        assert_eq!(
            ismethod_type(&[], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
    }

    #[test]
    fn metaclass_type_reports_unknown_metadata() {
        assert_eq!(
            metaclass_type(&[], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }
}

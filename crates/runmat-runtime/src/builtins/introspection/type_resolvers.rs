use runmat_builtins::{ResolveContext, Type};

pub fn ismethod_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

pub fn metaclass_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

pub fn which_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Union(vec![Type::String, Type::cell_of(Type::String)])
}

pub fn clearvars_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![Some(0), Some(0)]),
    }
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

    #[test]
    fn which_type_reports_union() {
        assert_eq!(
            which_type(&[], &ResolveContext::new(Vec::new())),
            Type::Union(vec![Type::String, Type::cell_of(Type::String)])
        );
    }

    #[test]
    fn clearvars_type_reports_empty_tensor() {
        assert_eq!(
            clearvars_type(&[], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![Some(0), Some(0)])
            }
        );
    }
}

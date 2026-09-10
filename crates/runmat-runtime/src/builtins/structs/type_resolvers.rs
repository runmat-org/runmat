use runmat_builtins::{ResolveContext, Type};

pub fn getfield_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

pub fn setfield_type(args: &[Type], _context: &ResolveContext) -> Type {
    args.first()
        .and_then(struct_container_type)
        .map(drop_struct_fields)
        .unwrap_or(Type::Unknown)
}

fn struct_container_type(ty: &Type) -> Option<Type> {
    match ty {
        Type::Struct { known_fields } => Some(Type::Struct {
            known_fields: known_fields.clone(),
        }),
        _ => None,
    }
}

fn drop_struct_fields(ty: Type) -> Type {
    match ty {
        Type::Struct { .. } => Type::Struct { known_fields: None },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::ResolveContext;

    #[test]
    fn getfield_type_is_unknown() {
        assert_eq!(
            getfield_type(&[], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }

    #[test]
    fn setfield_type_drops_known_fields() {
        assert_eq!(
            setfield_type(
                &[Type::Struct {
                    known_fields: Some(vec!["a".to_string()])
                }],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Struct { known_fields: None }
        );
    }
}

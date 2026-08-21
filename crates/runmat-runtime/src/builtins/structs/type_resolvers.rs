use runmat_builtins::{ResolveContext, Type};

pub fn fieldnames_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::cell_of(Type::String)
}

pub fn getfield_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

pub fn isfield_type(args: &[Type], _context: &ResolveContext) -> Type {
    let Some(name_type) = args.get(1) else {
        return Type::Unknown;
    };

    match name_type {
        Type::String => Type::Bool,
        Type::Cell { .. } => Type::logical(),
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn orderfields_type(args: &[Type], _context: &ResolveContext) -> Type {
    args.first()
        .and_then(struct_container_type)
        .unwrap_or(Type::Unknown)
}

pub fn rmfield_type(args: &[Type], _context: &ResolveContext) -> Type {
    args.first()
        .and_then(struct_container_type)
        .map(drop_struct_fields)
        .unwrap_or(Type::Unknown)
}

pub fn setfield_type(args: &[Type], _context: &ResolveContext) -> Type {
    args.first()
        .and_then(struct_container_type)
        .map(drop_struct_fields)
        .unwrap_or(Type::Unknown)
}

pub fn structfun_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Unknown
}

fn struct_container_type(ty: &Type) -> Option<Type> {
    match ty {
        Type::Struct { known_fields } => Some(Type::Struct {
            known_fields: known_fields.clone(),
        }),
        Type::Cell {
            element_type: Some(element_type),
            ..
        } => match &**element_type {
            Type::Struct { known_fields } => Some(Type::cell_of(Type::Struct {
                known_fields: known_fields.clone(),
            })),
            _ => None,
        },
        _ => None,
    }
}

fn drop_struct_fields(ty: Type) -> Type {
    match ty {
        Type::Struct { .. } => Type::Struct { known_fields: None },
        Type::Cell {
            element_type: Some(element_type),
            ..
        } => match *element_type {
            Type::Struct { .. } => Type::cell_of(Type::Struct { known_fields: None }),
            other => Type::Cell {
                element_type: Some(Box::new(other)),
                length: None,
            },
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::ResolveContext;

    #[test]
    fn fieldnames_type_is_string_cell() {
        assert_eq!(
            fieldnames_type(&[], &ResolveContext::new(Vec::new())),
            Type::cell_of(Type::String)
        );
    }

    #[test]
    fn getfield_type_is_unknown() {
        assert_eq!(
            getfield_type(&[], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }

    #[test]
    fn isfield_type_string_scalar_returns_bool() {
        assert_eq!(
            isfield_type(
                &[Type::Struct { known_fields: None }, Type::String],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Bool
        );
    }

    #[test]
    fn orderfields_type_preserves_struct() {
        assert_eq!(
            orderfields_type(
                &[Type::Struct { known_fields: None }],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Struct { known_fields: None }
        );
    }

    #[test]
    fn rmfield_type_preserves_struct_array_container() {
        assert_eq!(
            rmfield_type(
                &[Type::cell_of(Type::Struct { known_fields: None })],
                &ResolveContext::new(Vec::new()),
            ),
            Type::cell_of(Type::Struct { known_fields: None })
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

    #[test]
    fn structfun_type_is_unknown() {
        assert_eq!(
            structfun_type(&[], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }
}

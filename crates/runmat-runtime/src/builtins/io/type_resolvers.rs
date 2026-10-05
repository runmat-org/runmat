use runmat_builtins::{ResolveContext, Type};

pub(crate) fn string_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::String
}

pub(crate) fn struct_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Struct { known_fields: None }
}

pub fn data_unknown_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

use runmat_builtins::{ResolveContext, Type};

pub(crate) fn string_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::String
}

pub(crate) fn num_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Num
}

pub(crate) fn tensor_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::tensor()
}

pub(crate) fn struct_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Struct { known_fields: None }
}

pub fn accept_type(args: &[Type], ctx: &ResolveContext) -> Type {
    struct_type(args, ctx)
}

pub fn close_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn tcpserver_type(args: &[Type], ctx: &ResolveContext) -> Type {
    struct_type(args, ctx)
}


pub fn data_unknown_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::ResolveContext;

    macro_rules! assert_resolver {
        ($name:ident, $resolver:expr, $args:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let ctx = ResolveContext::new(Vec::new());
                assert_eq!($resolver($args, &ctx), $expected);
            }
        };
    }

    assert_resolver!(
        accept_type_resolver,
        accept_type,
        &[],
        Type::Struct { known_fields: None }
    );
    assert_resolver!(close_type_resolver, close_type, &[], Type::Num);
    assert_resolver!(
        tcpserver_type_resolver,
        tcpserver_type,
        &[],
        Type::Struct { known_fields: None }
    );
}

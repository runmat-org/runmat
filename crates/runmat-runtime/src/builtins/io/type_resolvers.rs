use runmat_builtins::{ResolveContext, Type};

pub(crate) fn string_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::String
}

pub(crate) fn num_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Num
}

pub(crate) fn bool_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Bool
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

pub fn read_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![Type::String, Type::tensor()])
}

pub fn readline_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![Type::String, Type::tensor()])
}

pub fn tcpclient_type(args: &[Type], ctx: &ResolveContext) -> Type {
    struct_type(args, ctx)
}

pub fn tcpserver_type(args: &[Type], ctx: &ResolveContext) -> Type {
    struct_type(args, ctx)
}

pub fn write_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn system_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Union(vec![
        Type::Num,
        Type::OutputList(vec![Type::Num, Type::String]),
    ])
}

pub fn readmatrix_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![Type::tensor(), Type::logical()])
}

pub fn xlsread_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![
        Type::tensor(),
        Type::OutputList(vec![Type::tensor(), Type::cell()]),
        Type::OutputList(vec![Type::tensor(), Type::cell(), Type::cell()]),
    ])
}

pub fn xlswrite_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![
        Type::Bool,
        Type::OutputList(vec![Type::Bool]),
        Type::OutputList(vec![Type::Bool, Type::Struct { known_fields: None }]),
    ])
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
        read_type_resolver,
        read_type,
        &[],
        Type::Union(vec![Type::String, Type::tensor()])
    );
    assert_resolver!(
        readline_type_resolver,
        readline_type,
        &[],
        Type::Union(vec![Type::String, Type::tensor()])
    );
    assert_resolver!(
        tcpclient_type_resolver,
        tcpclient_type,
        &[],
        Type::Struct { known_fields: None }
    );
    assert_resolver!(
        tcpserver_type_resolver,
        tcpserver_type,
        &[],
        Type::Struct { known_fields: None }
    );
    assert_resolver!(write_type_resolver, write_type, &[], Type::Num);

    assert_resolver!(csvread_type_resolver, tensor_type, &[], Type::tensor());
    assert_resolver!(csvwrite_type_resolver, num_type, &[], Type::Num);
    assert_resolver!(dlmread_type_resolver, tensor_type, &[], Type::tensor());
    assert_resolver!(dlmwrite_type_resolver, num_type, &[], Type::Num);
    assert_resolver!(
        readmatrix_type_resolver,
        readmatrix_type,
        &[],
        Type::Union(vec![Type::tensor(), Type::logical()])
    );
    assert_resolver!(writecell_type_resolver, num_type, &[], Type::Num);
    assert_resolver!(writematrix_type_resolver, num_type, &[], Type::Num);
    assert_resolver!(
        xlswrite_type_resolver,
        xlswrite_type,
        &[],
        Type::Union(vec![
            Type::Bool,
            Type::OutputList(vec![Type::Bool]),
            Type::OutputList(vec![Type::Bool, Type::Struct { known_fields: None },]),
        ])
    );
}

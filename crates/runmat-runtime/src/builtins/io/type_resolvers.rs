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

pub(crate) fn cell_struct_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::cell_of(Type::Struct { known_fields: None })
}

pub fn disp_type(args: &[Type], ctx: &ResolveContext) -> Type {
    tensor_type(args, ctx)
}

pub fn input_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![Type::String, Type::Num, Type::tensor()])
}

pub fn fclose_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn feof_type(args: &[Type], ctx: &ResolveContext) -> Type {
    bool_type(args, ctx)
}

pub fn fgets_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![Type::String, Type::Num])
}

pub fn fileread_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn filewrite_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn fopen_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::tensor();
    }
    match args.first() {
        Some(Type::Num) | Some(Type::Int) => Type::String,
        Some(Type::String) => Type::Union(vec![Type::Num, Type::tensor()]),
        Some(Type::Unknown) => Type::Union(vec![Type::String, Type::Num, Type::tensor()]),
        _ => Type::Union(vec![Type::Num, Type::tensor()]),
    }
}

pub fn fprintf_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn fread_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![Type::String, Type::tensor(), Type::logical()])
}

pub fn fwrite_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn weboptions_type(args: &[Type], ctx: &ResolveContext) -> Type {
    struct_type(args, ctx)
}

pub fn webread_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![
        Type::Struct { known_fields: None },
        Type::cell(),
        Type::String,
        Type::tensor(),
        Type::logical(),
        Type::Num,
        Type::Bool,
    ])
}

pub fn webwrite_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![
        Type::Struct { known_fields: None },
        Type::cell(),
        Type::String,
        Type::tensor(),
        Type::logical(),
        Type::Num,
        Type::Bool,
    ])
}

pub fn jsondecode_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![
        Type::Struct { known_fields: None },
        Type::cell(),
        Type::String,
        Type::tensor(),
        Type::logical(),
        Type::Num,
        Type::Bool,
    ])
}

pub fn jsonencode_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn load_type(args: &[Type], ctx: &ResolveContext) -> Type {
    struct_type(args, ctx)
}

pub fn save_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
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

pub fn addpath_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn cd_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn copyfile_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn delete_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn dir_type(args: &[Type], ctx: &ResolveContext) -> Type {
    cell_struct_type(args, ctx)
}

pub fn exist_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn genpath_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn fullfile_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn getenv_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return struct_type(args, _ctx);
    }
    match args.first() {
        Some(Type::Cell { .. }) => Type::cell_of(Type::String),
        Some(Type::String) => Type::String,
        Some(Type::Unknown) => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn ls_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn mkdir_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn movefile_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn path_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn pwd_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn rmdir_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn rmpath_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn savepath_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn setenv_type(args: &[Type], ctx: &ResolveContext) -> Type {
    num_type(args, ctx)
}

pub fn tempdir_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn tempname_type(args: &[Type], ctx: &ResolveContext) -> Type {
    string_type(args, ctx)
}

pub fn readmatrix_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let _ = args;
    Type::Union(vec![Type::tensor(), Type::logical()])
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

    assert_resolver!(disp_type_resolver, disp_type, &[], Type::tensor());
    assert_resolver!(
        input_type_resolver,
        input_type,
        &[],
        Type::Union(vec![Type::String, Type::Num, Type::tensor()])
    );

    assert_resolver!(fclose_type_resolver, fclose_type, &[], Type::Num);
    assert_resolver!(feof_type_resolver, feof_type, &[], Type::Bool);
    assert_resolver!(
        fgets_type_resolver,
        fgets_type,
        &[],
        Type::Union(vec![Type::String, Type::Num])
    );
    assert_resolver!(fileread_type_resolver, fileread_type, &[], Type::String);
    assert_resolver!(filewrite_type_resolver, filewrite_type, &[], Type::Num);
    assert_resolver!(fopen_type_resolver, fopen_type, &[], Type::tensor());
    assert_resolver!(fprintf_type_resolver, fprintf_type, &[], Type::Num);
    assert_resolver!(
        fread_type_resolver,
        fread_type,
        &[],
        Type::Union(vec![Type::String, Type::tensor(), Type::logical()])
    );
    assert_resolver!(fwrite_type_resolver, fwrite_type, &[], Type::Num);

    assert_resolver!(
        weboptions_type_resolver,
        weboptions_type,
        &[],
        Type::Struct { known_fields: None }
    );
    assert_resolver!(
        webread_type_resolver,
        webread_type,
        &[],
        Type::Union(vec![
            Type::Struct { known_fields: None },
            Type::cell(),
            Type::String,
            Type::tensor(),
            Type::logical(),
            Type::Num,
            Type::Bool,
        ])
    );
    assert_resolver!(
        webwrite_type_resolver,
        webwrite_type,
        &[],
        Type::Union(vec![
            Type::Struct { known_fields: None },
            Type::cell(),
            Type::String,
            Type::tensor(),
            Type::logical(),
            Type::Num,
            Type::Bool,
        ])
    );

    assert_resolver!(
        jsondecode_type_resolver,
        jsondecode_type,
        &[],
        Type::Union(vec![
            Type::Struct { known_fields: None },
            Type::cell(),
            Type::String,
            Type::tensor(),
            Type::logical(),
            Type::Num,
            Type::Bool,
        ])
    );
    assert_resolver!(jsonencode_type_resolver, jsonencode_type, &[], Type::String);

    assert_resolver!(
        load_type_resolver,
        load_type,
        &[],
        Type::Struct { known_fields: None }
    );
    assert_resolver!(save_type_resolver, save_type, &[], Type::Num);

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

    assert_resolver!(addpath_type_resolver, addpath_type, &[], Type::String);
    assert_resolver!(cd_type_resolver, cd_type, &[], Type::String);
    assert_resolver!(copyfile_type_resolver, copyfile_type, &[], Type::Num);
    assert_resolver!(delete_type_resolver, delete_type, &[], Type::Num);
    assert_resolver!(
        dir_type_resolver,
        dir_type,
        &[],
        Type::cell_of(Type::Struct { known_fields: None })
    );
    assert_resolver!(exist_type_resolver, exist_type, &[], Type::Num);
    assert_resolver!(genpath_type_resolver, genpath_type, &[], Type::String);
    assert_resolver!(fullfile_type_resolver, fullfile_type, &[], Type::String);
    assert_resolver!(
        getenv_type_resolver,
        getenv_type,
        &[],
        Type::Struct { known_fields: None }
    );
    assert_resolver!(ls_type_resolver, ls_type, &[], Type::String);
    assert_resolver!(mkdir_type_resolver, mkdir_type, &[], Type::Num);
    assert_resolver!(movefile_type_resolver, movefile_type, &[], Type::Num);
    assert_resolver!(path_type_resolver, path_type, &[], Type::String);
    assert_resolver!(pwd_type_resolver, pwd_type, &[], Type::String);
    assert_resolver!(rmdir_type_resolver, rmdir_type, &[], Type::Num);
    assert_resolver!(rmpath_type_resolver, rmpath_type, &[], Type::String);
    assert_resolver!(savepath_type_resolver, savepath_type, &[], Type::Num);
    assert_resolver!(setenv_type_resolver, setenv_type, &[], Type::Num);
    assert_resolver!(tempdir_type_resolver, tempdir_type, &[], Type::String);
    assert_resolver!(tempname_type_resolver, tempname_type, &[], Type::String);

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
    assert_resolver!(writematrix_type_resolver, num_type, &[], Type::Num);
}

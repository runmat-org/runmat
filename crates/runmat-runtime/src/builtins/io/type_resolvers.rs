use runmat_builtins::{ResolveContext, Type};

pub(crate) fn struct_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Struct { known_fields: None }
}

pub(crate) fn spreadsheet_import_options_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Object {
        class_name: Some(
            crate::builtins::io::tabular::import_options::SPREADSHEET_IMPORT_CLASS.into(),
        ),
        shape: Some(vec![Some(1), Some(1)]),
    }
}

pub(crate) fn detected_import_options_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Object {
        class_name: None,
        shape: Some(vec![Some(1), Some(1)]),
    }
}

pub fn data_unknown_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

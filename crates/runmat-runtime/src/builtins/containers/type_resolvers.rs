use runmat_builtins::Type;

pub fn map_handle_type(_args: &[Type]) -> Type {
    Type::Unknown
}

pub fn map_cell_type(_args: &[Type]) -> Type {
    Type::cell()
}

pub fn map_is_key_type(_args: &[Type]) -> Type {
    Type::logical()
}

pub fn map_unknown_type(_args: &[Type]) -> Type {
    Type::Unknown
}

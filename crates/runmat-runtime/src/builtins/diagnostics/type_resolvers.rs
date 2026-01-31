use runmat_builtins::Type;

pub fn assert_type(_args: &[Type]) -> Type {
    Type::Num
}

pub fn warning_type(_args: &[Type]) -> Type {
    Type::Unknown
}

pub fn error_type(_args: &[Type]) -> Type {
    Type::Unknown
}

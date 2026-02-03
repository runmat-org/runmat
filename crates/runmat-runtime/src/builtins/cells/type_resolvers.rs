use runmat_builtins::Type;

pub fn cell_type(_args: &[Type]) -> Type {
    Type::cell()
}

pub fn cell2mat_type(args: &[Type]) -> Type {
    match args.first() {
        Some(Type::Cell {
            element_type: Some(element_type),
            ..
        }) => cell2mat_element_type(element_type),
        Some(Type::Cell { .. }) => Type::Unknown,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

pub fn cellfun_type(_args: &[Type]) -> Type {
    Type::Unknown
}

pub fn cellstr_type(_args: &[Type]) -> Type {
    Type::cell_of(Type::String)
}

pub fn mat2cell_type(_args: &[Type]) -> Type {
    Type::cell()
}

fn cell2mat_element_type(element_type: &Type) -> Type {
    match element_type {
        Type::Bool | Type::Logical { .. } => Type::logical(),
        Type::Num | Type::Int | Type::Tensor { .. } => Type::tensor(),
        Type::String => Type::cell_of(Type::String),
        Type::Cell { .. }
        | Type::Struct { .. }
        | Type::Function { .. }
        | Type::Void
        | Type::Unknown
        | Type::Union(_) => Type::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_type_returns_cell() {
        assert_eq!(cell_type(&[]), Type::cell());
    }

    #[test]
    fn cell2mat_type_numeric_cells_return_tensor() {
        assert_eq!(cell2mat_type(&[Type::cell_of(Type::Num)]), Type::tensor());
    }

    #[test]
    fn cellfun_type_is_unknown() {
        assert_eq!(cellfun_type(&[]), Type::Unknown);
    }

    #[test]
    fn cellstr_type_is_string_cell() {
        assert_eq!(cellstr_type(&[Type::String]), Type::cell_of(Type::String));
    }

    #[test]
    fn mat2cell_type_is_cell() {
        assert_eq!(mat2cell_type(&[Type::tensor()]), Type::cell());
    }
}

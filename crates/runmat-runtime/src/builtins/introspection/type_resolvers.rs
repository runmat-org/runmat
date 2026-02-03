use runmat_builtins::Type;

pub fn class_type(_args: &[Type]) -> Type {
    Type::String
}

pub fn isa_type(_args: &[Type]) -> Type {
    Type::Bool
}

pub fn ischar_type(_args: &[Type]) -> Type {
    Type::Bool
}

pub fn isstring_type(_args: &[Type]) -> Type {
    Type::Bool
}

pub fn which_type(_args: &[Type]) -> Type {
    Type::Union(vec![Type::String, Type::cell_of(Type::String)])
}

pub fn who_type(_args: &[Type]) -> Type {
    Type::cell_of(Type::String)
}

pub fn whos_type(_args: &[Type]) -> Type {
    Type::cell_of(Type::Struct {
        known_fields: Some(vec![
            "name".to_string(),
            "size".to_string(),
            "bytes".to_string(),
            "class".to_string(),
            "global".to_string(),
            "sparse".to_string(),
            "complex".to_string(),
            "nesting".to_string(),
            "persistent".to_string(),
        ]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_type_reports_string() {
        assert_eq!(class_type(&[]), Type::String);
    }

    #[test]
    fn isa_type_reports_bool() {
        assert_eq!(isa_type(&[]), Type::Bool);
    }

    #[test]
    fn ischar_type_reports_bool() {
        assert_eq!(ischar_type(&[]), Type::Bool);
    }

    #[test]
    fn isstring_type_reports_bool() {
        assert_eq!(isstring_type(&[]), Type::Bool);
    }

    #[test]
    fn which_type_reports_union() {
        assert_eq!(
            which_type(&[]),
            Type::Union(vec![Type::String, Type::cell_of(Type::String)])
        );
    }

    #[test]
    fn who_type_reports_cell_of_strings() {
        assert_eq!(who_type(&[]), Type::cell_of(Type::String));
    }

    #[test]
    fn whos_type_reports_cell_of_structs() {
        assert_eq!(
            whos_type(&[]),
            Type::cell_of(Type::Struct {
                known_fields: Some(vec![
                    "name".to_string(),
                    "size".to_string(),
                    "bytes".to_string(),
                    "class".to_string(),
                    "global".to_string(),
                    "sparse".to_string(),
                    "complex".to_string(),
                    "nesting".to_string(),
                    "persistent".to_string(),
                ])
            })
        );
    }
}

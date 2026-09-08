use runmat_builtins::{ResolveContext, Type};

pub fn gpudevice_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Struct {
        known_fields: Some(vec![
            "backend".to_string(),
            "device_id".to_string(),
            "index".to_string(),
            "memory_bytes".to_string(),
            "name".to_string(),
            "precision".to_string(),
            "supports_double".to_string(),
            "vendor".to_string(),
        ]),
    }
}

pub fn gpuinfo_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::String
}

pub fn pagefun_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

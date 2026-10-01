use runmat_builtins::{ResolveContext, Type};

pub fn rng_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Struct {
        known_fields: Some(vec![
            "Seed".to_string(),
            "State".to_string(),
            "Type".to_string(),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_type_is_struct() {
        assert!(matches!(
            rng_type(&[], &ResolveContext::new(Vec::new())),
            Type::Struct { .. }
        ));
    }
}

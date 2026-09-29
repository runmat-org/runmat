use runmat_builtins::{ResolveContext, Type};

pub fn step_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn stepinfo_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Struct {
        known_fields: Some(vec![
            "Overshoot".to_string(),
            "Peak".to_string(),
            "PeakTime".to_string(),
            "RiseTime".to_string(),
            "SettlingMax".to_string(),
            "SettlingMin".to_string(),
            "SettlingTime".to_string(),
            "SteadyStateValue".to_string(),
            "TransientTime".to_string(),
            "Undershoot".to_string(),
        ]),
    }
}

pub fn impulse_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

pub fn nyquist_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::tensor()
}

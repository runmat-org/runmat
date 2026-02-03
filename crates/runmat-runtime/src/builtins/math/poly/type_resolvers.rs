use runmat_builtins::Type;

pub fn polyder_type(_args: &[Type]) -> Type {
    Type::tensor()
}

pub fn polyfit_type(_args: &[Type]) -> Type {
    Type::tensor()
}

pub fn polyint_type(_args: &[Type]) -> Type {
    Type::tensor()
}

pub fn polyval_type(_args: &[Type]) -> Type {
    Type::tensor()
}

pub fn roots_type(_args: &[Type]) -> Type {
    Type::tensor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyder_type_reports_tensor() {
        assert_eq!(polyder_type(&[]), Type::tensor());
    }

    #[test]
    fn polyfit_type_reports_tensor() {
        assert_eq!(polyfit_type(&[]), Type::tensor());
    }

    #[test]
    fn polyint_type_reports_tensor() {
        assert_eq!(polyint_type(&[]), Type::tensor());
    }

    #[test]
    fn polyval_type_reports_tensor() {
        assert_eq!(polyval_type(&[]), Type::tensor());
    }

    #[test]
    fn roots_type_reports_tensor() {
        assert_eq!(roots_type(&[]), Type::tensor());
    }
}

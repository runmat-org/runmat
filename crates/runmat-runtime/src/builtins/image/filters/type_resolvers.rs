use runmat_builtins::Type;

pub fn filter2_type(_args: &[Type]) -> Type {
    Type::tensor()
}

pub fn fspecial_type(_args: &[Type]) -> Type {
    Type::tensor()
}

pub fn imfilter_type(_args: &[Type]) -> Type {
    Type::tensor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter2_type_reports_tensor() {
        assert_eq!(filter2_type(&[]), Type::tensor());
    }

    #[test]
    fn fspecial_type_reports_tensor() {
        assert_eq!(fspecial_type(&[]), Type::tensor());
    }

    #[test]
    fn imfilter_type_reports_tensor() {
        assert_eq!(imfilter_type(&[]), Type::tensor());
    }
}

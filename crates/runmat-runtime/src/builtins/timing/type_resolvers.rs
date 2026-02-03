use runmat_builtins::Type;

pub fn pause_type(_args: &[Type]) -> Type {
    Type::Union(vec![Type::String, Type::tensor()])
}

pub fn tic_type(_args: &[Type]) -> Type {
    Type::Num
}

pub fn timeit_type(_args: &[Type]) -> Type {
    Type::Num
}

pub fn toc_type(_args: &[Type]) -> Type {
    Type::Num
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_type_reports_union() {
        assert_eq!(
            pause_type(&[]),
            Type::Union(vec![Type::String, Type::tensor()])
        );
    }

    #[test]
    fn tic_type_reports_num() {
        assert_eq!(tic_type(&[]), Type::Num);
    }

    #[test]
    fn timeit_type_reports_num() {
        assert_eq!(timeit_type(&[]), Type::Num);
    }

    #[test]
    fn toc_type_reports_num() {
        assert_eq!(toc_type(&[]), Type::Num);
    }
}

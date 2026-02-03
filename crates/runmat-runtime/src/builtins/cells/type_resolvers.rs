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

pub fn mat2cell_type(args: &[Type]) -> Type {
    let Some(input) = args.first() else {
        return Type::cell();
    };

    match mat2cell_element_type(input) {
        Some(element_type) => Type::cell_of(element_type),
        None => Type::cell(),
    }
}

fn cell2mat_element_type(element_type: &Type) -> Type {
    match element_type {
        Type::Union(options) => {
            let mut mapped: Vec<Type> = Vec::new();
            for option in options {
                let resolved = cell2mat_element_type(option);
                if matches!(resolved, Type::Unknown) {
                    return Type::Unknown;
                }
                push_unique_union(&mut mapped, resolved);
            }
            squash_union(mapped)
        }
        Type::Bool | Type::Logical { .. } => Type::logical(),
        Type::Num | Type::Int | Type::Tensor { .. } => Type::tensor(),
        Type::String => Type::cell_of(Type::String),
        Type::Cell { .. }
        | Type::Struct { .. }
        | Type::Function { .. }
        | Type::Void
        | Type::Unknown => Type::Unknown,
    }
}

fn mat2cell_element_type(input: &Type) -> Option<Type> {
    match input {
        Type::Union(options) => {
            let mut mapped: Vec<Type> = Vec::new();
            for option in options {
                let resolved = mat2cell_element_type(option)?;
                push_unique_union(&mut mapped, resolved);
            }
            if mapped.is_empty() {
                None
            } else {
                Some(squash_union(mapped))
            }
        }
        Type::Tensor { .. } | Type::Num | Type::Int | Type::Bool => {
            Some(Type::Union(vec![Type::Num, Type::tensor()]))
        }
        Type::Logical { .. } => Some(Type::Union(vec![Type::Bool, Type::logical()])),
        Type::String => Some(Type::Union(vec![Type::String, Type::cell_of(Type::String)])),
        Type::Cell {
            element_type: Some(element_type),
            ..
        } if matches!(**element_type, Type::String) => {
            Some(Type::Union(vec![Type::String, Type::cell_of(Type::String)]))
        }
        _ => None,
    }
}

fn push_unique_union(target: &mut Vec<Type>, candidate: Type) {
    match candidate {
        Type::Union(inner) => {
            for entry in inner {
                push_unique_union(target, entry);
            }
        }
        other => {
            if !target.iter().any(|existing| existing == &other) {
                target.push(other);
            }
        }
    }
}

fn squash_union(mut options: Vec<Type>) -> Type {
    if options.len() == 1 {
        options.pop().unwrap()
    } else {
        Type::Union(options)
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
    fn cell2mat_type_union_numeric_returns_tensor() {
        assert_eq!(
            cell2mat_type(&[Type::cell_of(Type::Union(vec![Type::Num, Type::Int]))]),
            Type::tensor()
        );
    }

    #[test]
    fn cell2mat_type_union_mixed_returns_union() {
        assert_eq!(
            cell2mat_type(&[Type::cell_of(Type::Union(vec![Type::Num, Type::Bool]))]),
            Type::Union(vec![Type::tensor(), Type::logical()])
        );
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
        assert_eq!(
            mat2cell_type(&[Type::tensor()]),
            Type::cell_of(Type::Union(vec![Type::Num, Type::tensor()]))
        );
    }

    #[test]
    fn mat2cell_type_logical_returns_cell_union() {
        assert_eq!(
            mat2cell_type(&[Type::logical()]),
            Type::cell_of(Type::Union(vec![Type::Bool, Type::logical()]))
        );
    }

    #[test]
    fn mat2cell_type_string_returns_cell_union() {
        assert_eq!(
            mat2cell_type(&[Type::String]),
            Type::cell_of(Type::Union(vec![Type::String, Type::cell_of(Type::String)]))
        );
    }
}

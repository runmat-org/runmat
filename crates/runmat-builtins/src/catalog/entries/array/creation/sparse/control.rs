/// Public sparse storage spelling, decoded once at the argument boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseStorageType {
    Double,
    Single,
    Logical,
}

/// Constructor form selected by admitted arity and whether argument three is
/// text. Unknown argument contents do not determine dimensions or stored nnz.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseConstructionForm {
    Conversion,
    Dimensions,
    Triplets,
}
impl SparseConstructionForm {
    pub fn for_arguments(count: usize, third_is_text: bool) -> Option<Self> {
        match count {
            1 => Some(Self::Conversion),
            2 => Some(Self::Dimensions),
            3 if third_is_text => Some(Self::Dimensions),
            3 | 5 | 6 => Some(Self::Triplets),
            _ => None,
        }
    }
}
impl SparseStorageType {
    pub fn parse(label: &str) -> Option<Self> {
        match label.to_ascii_lowercase().as_str() {
            "double" => Some(Self::Double),
            "single" => Some(Self::Single),
            "logical" => Some(Self::Logical),
            _ => None,
        }
    }
}

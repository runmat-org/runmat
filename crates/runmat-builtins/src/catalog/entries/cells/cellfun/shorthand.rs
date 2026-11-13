#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CellfunShorthand {
    IsClass,
    IsEmpty,
    IsLogical,
    IsReal,
    Length,
    Ndims,
    ProdOfSize,
    Size,
}

impl CellfunShorthand {
    pub(super) fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "isclass" => Some(Self::IsClass),
            "isempty" => Some(Self::IsEmpty),
            "islogical" => Some(Self::IsLogical),
            "isreal" => Some(Self::IsReal),
            "length" => Some(Self::Length),
            "ndims" => Some(Self::Ndims),
            "prodofsize" => Some(Self::ProdOfSize),
            "size" => Some(Self::Size),
            _ => None,
        }
    }

    pub(super) fn returns_logical(self) -> bool {
        matches!(
            self,
            Self::IsClass | Self::IsEmpty | Self::IsLogical | Self::IsReal
        )
    }
}

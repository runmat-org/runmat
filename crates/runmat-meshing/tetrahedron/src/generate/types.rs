#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetrahedronGenerationError {
    InvalidProtectedBoundaryComplex,
    EmptyProtectedBoundaryComplex,
    MissingPlcNode { node_id: String },
    NonFinitePlcNode { node_id: String },
    NonFiniteInteriorPoint,
    DegeneratePlcBounds,
    UnsupportedStructuredBoxPlc,
    DegenerateBoundaryFacet { facet_id: String },
}

impl std::fmt::Display for TetrahedronGenerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProtectedBoundaryComplex => {
                write!(formatter, "Tetrahedron generation requires a validated PLC")
            }
            Self::EmptyProtectedBoundaryComplex => {
                write!(formatter, "validated PLC has no nodes or facets")
            }
            Self::MissingPlcNode { node_id } => {
                write!(formatter, "PLC facet references missing node {node_id}")
            }
            Self::NonFinitePlcNode { node_id } => {
                write!(formatter, "PLC node {node_id} has non-finite coordinates")
            }
            Self::NonFiniteInteriorPoint => {
                write!(formatter, "PLC interior insertion point is non-finite")
            }
            Self::DegeneratePlcBounds => {
                write!(formatter, "validated PLC bounds are degenerate")
            }
            Self::UnsupportedStructuredBoxPlc => {
                write!(
                    formatter,
                    "validated PLC is not an axis-aligned structured box"
                )
            }
            Self::DegenerateBoundaryFacet { facet_id } => {
                write!(
                    formatter,
                    "PLC facet {facet_id} creates a degenerate Tetrahedron4"
                )
            }
        }
    }
}

impl std::error::Error for TetrahedronGenerationError {}

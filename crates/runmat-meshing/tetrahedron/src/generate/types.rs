use runmat_meshing_plc::validate::PlcValidationError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetrahedronGenerationError {
    InvalidProtectedBoundaryComplex {
        error: PlcValidationError,
    },
    MissingPlcNode {
        node_id: String,
    },
    NonFinitePlcNode {
        node_id: String,
    },
    NonFiniteInteriorPoint,
    DegeneratePlcBounds,
    UnsupportedStructuredBoxPlc,
    UnsupportedSingleTetrahedronPlc,
    UnsupportedNestedShellPlc {
        outer_shell_count: usize,
        nested_shell_count: usize,
        max_nesting_depth: usize,
    },
    UnsupportedNestedTetrahedronShellPlc,
    DegenerateNestedTetrahedronShellPlc,
    DegenerateSingleTetrahedronPlc,
    UnsupportedConvexPolyhedronPlc,
    DegenerateConvexPolyhedronPlc,
    UnsupportedHoledPolyhedronPlc,
    DegenerateHoledPolyhedronPlc,
    UnsupportedStarShapedPolyhedronPlc,
    DegenerateStarShapedPolyhedronPlc,
    DegenerateBoundaryFacet {
        facet_id: String,
    },
}

impl std::fmt::Display for TetrahedronGenerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProtectedBoundaryComplex { error } => {
                write!(
                    formatter,
                    "Tetrahedron generation requires a validated PLC: {error}"
                )
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
            Self::UnsupportedSingleTetrahedronPlc => {
                write!(
                    formatter,
                    "validated PLC is not a single Tetrahedron4 boundary"
                )
            }
            Self::UnsupportedNestedShellPlc {
                outer_shell_count,
                nested_shell_count,
                max_nesting_depth,
            } => write!(
                formatter,
                "validated PLC has unsupported shell nesting: {outer_shell_count} outer shells, {nested_shell_count} nested shells, max nesting depth {max_nesting_depth}"
            ),
            Self::UnsupportedNestedTetrahedronShellPlc => {
                write!(
                    formatter,
                    "validated PLC is not a supported nested Tetrahedron shell"
                )
            }
            Self::DegenerateNestedTetrahedronShellPlc => {
                write!(
                    formatter,
                    "validated nested Tetrahedron shell PLC would create degenerate Tetrahedron4 elements"
                )
            }
            Self::DegenerateSingleTetrahedronPlc => {
                write!(formatter, "validated single Tetrahedron4 PLC is degenerate")
            }
            Self::UnsupportedConvexPolyhedronPlc => {
                write!(
                    formatter,
                    "validated PLC is not a supported convex triangulated polyhedron"
                )
            }
            Self::DegenerateConvexPolyhedronPlc => {
                write!(
                    formatter,
                    "validated convex polyhedron PLC would create degenerate Tetrahedron4 elements"
                )
            }
            Self::UnsupportedHoledPolyhedronPlc => {
                write!(
                    formatter,
                    "validated PLC is not a supported holed triangulated polyhedron"
                )
            }
            Self::DegenerateHoledPolyhedronPlc => {
                write!(
                    formatter,
                    "validated holed polyhedron PLC would create degenerate Tetrahedron4 elements"
                )
            }
            Self::UnsupportedStarShapedPolyhedronPlc => {
                write!(
                    formatter,
                    "validated PLC is not a supported star-shaped triangulated polyhedron"
                )
            }
            Self::DegenerateStarShapedPolyhedronPlc => {
                write!(
                    formatter,
                    "validated star-shaped polyhedron PLC would create degenerate Tetrahedron4 elements"
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

use super::*;

pub const DELAUNAY_TRI_CLASS: runmat_types::StaticClassIdentity =
    runmat_builtins::DELAUNAY_TRI_CLASS;
pub(super) const BUILTIN_NAME: &str = "DelaunayTri";
pub(super) const POINTS_PROPERTY: &str = "Points";
pub(super) const CONNECTIVITY_PROPERTY: &str = "ConnectivityList";
pub(super) const LEGACY_POINTS_PROPERTY: &str = "X";
pub(super) const LEGACY_CONNECTIVITY_PROPERTY: &str = "Triangulation";
pub(super) const CONSTRAINTS_PROPERTY: &str = "Constraints";
pub(super) const DIMENSION_PROPERTY: &str = "Dimension";

pub(super) const INTEGER_COORDINATES_EXTENSION: BuiltinExtensionDescriptor =
    runmat_builtins::DELAUNAY_TRI_INTEGER_COORDINATES_EXTENSION;
pub(super) const INTEGER_TOPOLOGY_EXTENSION: BuiltinExtensionDescriptor =
    runmat_builtins::DELAUNAY_TRI_INTEGER_TOPOLOGY_EXTENSION;
pub(super) const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor =
    runmat_builtins::DELAUNAY_TRI_ERROR_INVALID_ARGUMENT;
pub(super) const ERROR_UNSUPPORTED: BuiltinErrorDescriptor =
    runmat_builtins::DELAUNAY_TRI_ERROR_UNSUPPORTED;

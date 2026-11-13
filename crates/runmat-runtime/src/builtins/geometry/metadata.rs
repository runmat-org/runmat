//! Runtime-only payload property and canonical catalog contract reexports.
//! Language object class identities and descriptors are owned by the typed
//! geometry catalog leaves, not duplicated by the runtime binding.

pub use runmat_builtins::{
    GEOMETRY_ASSET_CLASS, GEOMETRY_INSPECT_DESCRIPTOR, GEOMETRY_INSPECT_RESULT_CLASS,
    GEOMETRY_LIST_REGIONS_DESCRIPTOR, GEOMETRY_LIST_REGIONS_INTEGER_AUDIT,
    GEOMETRY_LOAD_DESCRIPTOR, GEOMETRY_MESHES_DESCRIPTOR, GEOMETRY_MESHES_INTEGER_AUDIT,
};
pub(super) use runmat_builtins::{
    GEOMETRY_INSPECT_ERROR_INPUT, GEOMETRY_INSPECT_ERROR_INTERNAL, GEOMETRY_INSPECT_ERROR_IO,
    GEOMETRY_INSPECT_ERROR_OPERATION, GEOMETRY_INSPECT_ERROR_OUTPUTS,
    GEOMETRY_LIST_REGIONS_ERROR_OUTPUTS, GEOMETRY_LOAD_ERROR_INPUT, GEOMETRY_LOAD_ERROR_INTERNAL,
    GEOMETRY_LOAD_ERROR_IO, GEOMETRY_LOAD_ERROR_OPERATION, GEOMETRY_LOAD_ERROR_OUTPUTS,
    GEOMETRY_MESHES_ERROR_OUTPUTS,
};

pub const GEOMETRY_ASSET_JSON_PROPERTY: &str = "__runmat_geometry_asset_json";
pub(super) const GEOMETRY_LOAD_NAME: &str = "geometry.load";
pub(super) const GEOMETRY_INSPECT_NAME: &str = "geometry.inspect";
pub(super) const GEOMETRY_LIST_REGIONS_NAME: &str = "geometry.listRegions";
pub(super) const GEOMETRY_MESHES_NAME: &str = "geometry.meshes";

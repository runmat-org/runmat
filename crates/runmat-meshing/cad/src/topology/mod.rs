pub mod source;
pub(crate) mod source_mesh;

pub use source::*;
pub use source_mesh::{
    extract_source_topology, SourceTopologyEdge, SourceTopologyError, SourceTopologyFace,
    SourceTopologyModel, SourceTopologyVertex,
};

pub const MODULE_PURPOSE: &str =
    "normalized CAD vertices, edges, loops, faces, shells, and volumes";

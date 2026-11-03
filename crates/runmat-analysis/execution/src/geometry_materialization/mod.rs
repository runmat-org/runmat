mod contract;
mod generated_product;
mod generator;
mod output;
mod program;
mod source;
mod task;

pub use contract::*;
pub use generated_product::*;
pub use generator::*;
pub use output::*;
pub use program::*;
pub use source::*;
pub use task::*;

#[cfg(test)]
mod generator_tests;
#[cfg(test)]
mod source_tests;

mod host;
pub use host::{
    execute_geometry_materialization_program_request, occt_geometry_provider_profile,
    GeometryMaterializationLimits,
};

mod occt_evaluator;
pub use occt_evaluator::OcctMeshingEvaluatorProvider;
mod serial_mesh;
mod imported_mesh;
pub use imported_mesh::{
    mesh_imported_source_serial, ImportedMeshContactSelectors, ImportedSourceMeshingRequest,
    ImportedSourceMeshingResult,
};
pub use serial_mesh::{
    materialize_and_mesh_serial, PreparedFeaGeometry, SerialGeometryMeshingError,
    SerialGeometryMeshingResult,
};

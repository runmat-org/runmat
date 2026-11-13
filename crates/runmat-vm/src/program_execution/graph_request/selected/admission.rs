//! Borrowed admission views; neither route constructs the other's wire form.

use super::*;
use runmat_execution::value::{
    GraphClassDependency, GraphClassMethodDependency, GraphInvocationRoots, GraphStaticDependency,
    PayloadGraph,
};
use runmat_execution_artifact::ArchiveAdmittedCapturedGraphCode;

pub(super) enum Admission<'a> {
    Worker(&'a ArchiveAdmittedGraphExecutionRequest),
    Captured(&'a ArchiveAdmittedCapturedGraphCode),
}

pub(super) struct View<'a> {
    pub primary_unit: &'a [u8],
    pub retained_units: &'a [Vec<u8>],
    pub graph: &'a PayloadGraph,
    pub class_dependencies: &'a [GraphClassDependency],
    pub class_method_dependencies: &'a [GraphClassMethodDependency],
    pub static_dependencies: &'a [GraphStaticDependency],
    pub roots: Option<&'a GraphInvocationRoots>,
    pub primary_revision: Option<&'a ProgramRevision>,
    pub schema_version: u16,
}

impl Admission<'_> {
    pub(super) fn view(&self) -> View<'_> {
        match self {
            Self::Worker(admitted) => {
                let request = admitted.request();
                View {
                    primary_unit: &request.artifact.executable_bytes,
                    retained_units: &request.retained_units,
                    graph: &request.graph,
                    class_dependencies: &request.class_dependencies,
                    class_method_dependencies: &request.class_method_dependencies,
                    static_dependencies: &request.static_dependencies,
                    roots: Some(&request.roots),
                    primary_revision: Some(&request.recipe.program_revision),
                    schema_version: request.schema_version,
                }
            }
            Self::Captured(admitted) => {
                let code = admitted.code();
                View {
                    primary_unit: &code.primary_unit,
                    retained_units: &code.retained_units,
                    graph: &code.graph,
                    class_dependencies: &code.class_dependencies,
                    class_method_dependencies: &code.class_method_dependencies,
                    static_dependencies: &code.static_dependencies,
                    roots: None,
                    primary_revision: None,
                    schema_version: code.schema_version,
                }
            }
        }
    }

    pub(super) fn source_map(&self) -> Result<ExecutableSourceMapV2, String> {
        match self {
            Self::Worker(admitted) => admitted.request().selected_source_map(),
            Self::Captured(admitted) => admitted.code().selected_source_map(),
        }
        .map_err(|error| error.to_string())
    }

    pub(super) fn consume_bundle(
        &self,
        position: usize,
        ticket: AdmittedGraphUnitBundle,
    ) -> Result<Option<runmat_execution_artifact::ExecutionBundle>, String> {
        match self {
            Self::Worker(admitted) => admitted.consume_bundle_for(position, ticket),
            Self::Captured(admitted) => admitted.consume_bundle_for(position, ticket),
        }
        .map_err(|error| error.to_string())
    }
}

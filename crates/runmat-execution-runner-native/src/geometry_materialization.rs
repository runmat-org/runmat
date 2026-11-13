//! Native platform adapter for the shared exact geometry materialization host.
pub use runmat_analysis_execution::{
    occt_geometry_provider_profile as native_geometry_provider_profile,
    GeometryMaterializationLimits as NativeGeometryMaterializationLimits,
};
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};

pub fn execute_geometry_materialization_program_request<
    S: runmat_execution_artifact::cache::CacheImport + runmat_execution_artifact::cache::CacheExport,
>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    limits: NativeGeometryMaterializationLimits,
) -> ProgramExecutionResponse {
    let platform = crate::meshing_platform::native_meshing_platform(
        None,
        native_geometry_provider_profile().map(|profile| profile.kernel_abi),
    );
    runmat_analysis_execution::execute_geometry_materialization_program_request(
        request, store, limits, platform,
    )
}

#[cfg(all(test, feature = "occt-native"))]
mod geometry_materialization_tests;
#[cfg(all(test, feature = "occt-native"))]
mod geometry_sweep_tests;

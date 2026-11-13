//! Exact preparation in the browser sidecar uses the shared production host.
use runmat_analysis_execution::PreparedFeaGeometry;
use runmat_analysis_execution::{materialize_and_mesh_serial, occt_geometry_provider_profile};
use runmat_execution::Digest;
use runmat_execution_artifact::{
    cache::{CacheExport, MemoryObjectStore},
    object::ObjectInventoryLimits,
    LogicalObject, ProgramExecutionRequest,
};
use runmat_meshing_core::{MeshingRequestSettings, PlatformBuildIdentity, StableDigest};
use serde::Deserialize;

const MAX_REQUEST_BYTES: usize = 64 << 20;
const MAX_GEOMETRY_MEMORY_BYTES: u64 = 512 << 20;
const INVENTORY: ObjectInventoryLimits = ObjectInventoryLimits {
    max_objects: 100_000,
    max_object_bytes: 64 << 20,
    max_total_bytes: 128 << 20,
};
#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Profile,
    MaterializeAndMesh {
        program: Box<ProgramExecutionRequest>,
        objects: Vec<LogicalObject>,
        settings: MeshingRequestSettings,
    },
    MeshImportedSource {
        request: Box<runmat_analysis_execution::ImportedSourceMeshingRequest>,
    },
}
struct Cancellation;
impl runmat_meshing_core::MeshingCancellationSignal for Cancellation {
    fn is_cancelled(&self) -> bool {
        #[cfg(target_os = "emscripten")]
        unsafe {
            unsafe extern "C" {
                fn emscripten_run_script_int(script: *const std::ffi::c_char) -> std::ffi::c_int;
            }
            return emscripten_run_script_int(
                c"Boolean(globalThis.__runmatFeaCancelled?.())".as_ptr(),
            ) != 0;
        }
        #[cfg(not(target_os = "emscripten"))]
        false
    }
}
pub(super) fn execute(
    json: &str,
) -> Result<String, runmat_analysis_execution::SerialGeometryMeshingError> {
    if json.len() > MAX_REQUEST_BYTES {
        return Err("Geometry preparation exceeds this host's input allowance.".into());
    }
    let request: Request = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let profile = occt_geometry_provider_profile()
        .ok_or_else(|| "Exact CAD execution is unavailable.".to_owned())?;
    match request {
        Request::Profile => serde_json::to_string(&profile).map_err(|e| e.to_string().into()),
        Request::MeshImportedSource { request } => {
            // The surrounding owned-worker JS host validates the original
            // parent delegation before this call and before returning bytes.
            let ceilings = runmat_meshing_core::MeshingExecutionCeilings::for_host(
                MAX_GEOMETRY_MEMORY_BYTES,
                &request.settings.limits,
            );
            let platform = PlatformBuildIdentity {
                capability_cohort: "browser-serial-occt".into(),
                target_triple: "wasm32-unknown-emscripten".into(),
                build_digest: StableDigest::from_bytes(
                    *Digest::sha256(concat!("runmat-exact-browser:", env!("CARGO_PKG_VERSION")))
                        .bytes(),
                ),
                exact_kernel_abi: Some(profile.kernel_abi),
            };
            let mut store = MemoryObjectStore::new(INVENTORY).map_err(|error| error.to_string())?;
            let context =
                runmat_geometry_io::GeometryImportContext::with_cancellation_probe(|| {
                    runmat_meshing_core::MeshingCancellationSignal::is_cancelled(&Cancellation)
                });
            let result = runmat_analysis_execution::mesh_imported_source_serial(
                *request,
                &mut store,
                &context,
                ceilings,
                platform,
                INVENTORY,
                &Cancellation,
                &mut runmat_meshing_execution::NoopMeshingProgress,
            )?;
            serde_json::to_string(&result).map_err(|error| error.to_string().into())
        }
        Request::MaterializeAndMesh {
            program,
            objects,
            settings,
        } => {
            settings.validate().map_err(|e| e.to_string())?;
            // The browser host admits at most MAX_GEOMETRY_MEMORY_BYTES; a mesh that needs more
            // stops at this ceiling with a typed meshing resource failure.
            let ceilings = runmat_meshing_core::MeshingExecutionCeilings::for_host(
                MAX_GEOMETRY_MEMORY_BYTES,
                &settings.limits,
            );
            // Validate the complete transferred inventory before retaining copies.
            runmat_execution_artifact::object::validate_inventory(&objects, INVENTORY)
                .map_err(|e| e.to_string())?;
            let mut store = MemoryObjectStore::new(INVENTORY).map_err(|e| e.to_string())?;
            for object in objects {
                store.write_verified(&object).map_err(|e| e.to_string())?;
            }
            let platform = PlatformBuildIdentity {
                capability_cohort: "browser-serial-occt".into(),
                target_triple: "wasm32-unknown-emscripten".into(),
                build_digest: StableDigest::from_bytes(
                    *Digest::sha256(concat!("runmat-exact-browser:", env!("CARGO_PKG_VERSION")))
                        .bytes(),
                ),
                exact_kernel_abi: Some(profile.kernel_abi),
            };
            let completed = materialize_and_mesh_serial(
                &program,
                &mut store,
                settings,
                ceilings,
                platform,
                INVENTORY,
                &Cancellation,
                &mut runmat_meshing_execution::NoopMeshingProgress,
            )?;
            serde_json::to_string(&PreparedFeaGeometry {
                geometry: completed.geometry,
                mesh: completed.meshing.artifact,
                meshing_evidence: completed.meshing.evidence,
            })
            .map_err(|e| e.to_string().into())
        }
    }
}

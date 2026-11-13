//! Named targets share configuration, immutable compilation and existing run owners.
use runmat_core::project_entrypoints::CapturedProjectEntrypoints;
use std::path::Path;
use wasm_bindgen::prelude::*;

mod contracts;
mod listing;
mod local;
use contracts::*;

const FEA_CLUSTER_LIMITATION: &str = "Browser FEA cluster preparation has no portable typed-FEA geometry/mesh/artifact producer. The native OCCT-enabled CLI owns this route: runmat fea run <named-entrypoint> --cluster <cluster>, with admitted cluster access. Desktop caller integration is not established by this API.";

async fn capture(start: &Path) -> Result<CapturedProjectEntrypoints, JsValue> {
    CapturedProjectEntrypoints::capture(start)
        .await
        .map_err(|error| failure("project.entrypoint.capture", error))
}

fn path(target: &runmat_config::project::ResolvedProjectEntrypoint) -> Result<String, JsValue> {
    target
        .source_file
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| {
            failure(
                "project.entrypoint.path",
                "browser source path is not UTF-8",
            )
        })
}

// Selects the language checker, never readiness or producer capabilities.
fn is_fea(target: &runmat_config::project::ResolvedProjectEntrypoint) -> bool {
    target
        .source_file
        .extension()
        .is_some_and(|extension| extension == "fea")
}

fn cluster_policy(
    project: &CapturedProjectEntrypoints,
) -> Result<runmat_config::runtime::ClusterRuntimePolicy, JsValue> {
    runmat_config::runtime::ClusterRuntimePolicy::from_project_document(
        &project.project_document.source,
        project.project_document.format,
    )
    .map_err(|error| failure("project.entrypoint.policy", error))
}

#[wasm_bindgen(js_name = prepareNamedClusterProgram)]
pub async fn prepare_named_cluster_program(
    start: String,
    name: String,
    inputs: JsValue,
) -> Result<JsValue, JsValue> {
    let inputs = optional_inputs(inputs)?;
    let (scope, paths) =
        runmat_filesystem::CapturedReadScope::capture(&[Path::new(&start), Path::new(".")], inputs)
            .map_err(|error| failure("project.entrypoint.filesystem", error))?;
    scope
        .run(async {
            let project = capture(&paths[0]).await?;
            let target = project
                .resolve(&name)
                .await
                .map_err(|error| failure("project.entrypoint.resolve", error))?;
            let policy = cluster_policy(&project)?;
            if is_fea(&target) {
                let config = policy
                    .runtime_config()
                    .map_err(|error| failure("project.entrypoint.policy", error))?;
                let mut session = runmat_core::RunMatSession::with_options(false, false)
                    .map_err(|error| failure("project.entrypoint.session", error))?;
                session
                    .apply_cluster_runtime_config(&config)
                    .map_err(|error| failure("project.entrypoint.policy", error))?;
                session
                    .install_project_handoff(project.handoff.clone())
                    .map_err(|error| failure("project.entrypoint.install", error))?;
                session
                    .load_and_check_fea_source_path(&target.source_file)
                    .await
                    .map_err(|error| failure("project.entrypoint.check", error))?;
                return Err(failure(
                    "project.entrypoint.browserFeaClusterUnavailable",
                    FEA_CLUSTER_LIMITATION,
                ));
            }
            let source_path = path(&target)?;
            super::cluster_program::prepare_project_program(
                project.handoff,
                source_path,
                target.function,
                policy,
            )
            .await
        })
        .await
        .map_err(|error| failure("project.entrypoint.filesystem", error))?
}

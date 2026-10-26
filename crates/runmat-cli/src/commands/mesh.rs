use std::path::{Path, PathBuf};

use anyhow::Result;
use runmat_execution_runner_native::{
    mesh_source_with_native_execution, native_meshing_execution_provider,
    NativeMeshingSourceRequest,
};
use runmat_geometry_core::LengthUnit;
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{ElementOrder, MeshingRequestSettings};
use runmat_runtime::meshing::MeshingExecutionProvider;

use crate::cli::{ExecutionTargetArgs, MeshElementOrderArg, MeshSourceUnitsArg};
use crate::presentation;

mod persistence;
mod remote;
pub(crate) use persistence::MeshOutputSummary;
use persistence::{persist_completed_mesh, ExistingOutputPolicy};

pub(crate) use remote::materialize_outputs as materialize_remote_outputs;

pub struct MeshCommand {
    pub source: PathBuf,
    pub source_units: MeshSourceUnitsArg,
    pub output: Option<PathBuf>,
    pub evidence: Option<PathBuf>,
    pub target_size_m: f64,
    pub maximum_deviation_m: f64,
    pub element_order: MeshElementOrderArg,
    pub maximum_elements: u64,
    pub deterministic_seed: u64,
    pub shared_interfaces: bool,
    pub force: bool,
    pub json: bool,
}

pub fn execute(command: MeshCommand) -> Result<()> {
    execute_local(command)
}

pub async fn execute_with_target(
    command: MeshCommand,
    execution: ExecutionTargetArgs,
    cli: &crate::cli::Cli,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
) -> Result<()> {
    match execution.resolve()? {
        crate::cli::ExecutionTarget::Local => execute_local(command),
        crate::cli::ExecutionTarget::Remote(target) => {
            remote::execute(command, target, cli, config).await
        }
    }
}

fn execute_local(command: MeshCommand) -> Result<()> {
    let output = command
        .output
        .clone()
        .unwrap_or_else(|| command.source.with_extension("solver-mesh.cbor"));
    let evidence = command
        .evidence
        .clone()
        .unwrap_or_else(|| command.source.with_extension("meshing-evidence.cbor"));
    validate_output_paths(&output, &evidence, command.force)?;

    let (artifact, mesh_evidence) = mesh_source_with_native_execution(NativeMeshingSourceRequest {
        source: command.source.clone(),
        import_options: import_options(&command),
        contact_definitions: Vec::new(),
        expected_source_sha256: None,
        settings: meshing_settings(&command),
        partition_policy: Default::default(),
    })
    .map(|generated| (generated.dag_result.artifact, generated.dag_result.evidence))
    .map_err(anyhow::Error::new)?;

    let summary = persist_completed_mesh(
        &output,
        &evidence,
        artifact,
        mesh_evidence,
        if command.force {
            ExistingOutputPolicy::Replace
        } else {
            ExistingOutputPolicy::CreateNew
        },
    )?;
    print_result(&command, &output, &evidence, &summary)
}

fn meshing_settings(command: &MeshCommand) -> MeshingRequestSettings {
    let mut settings = MeshingRequestSettings {
        element_order: match command.element_order {
            MeshElementOrderArg::Tet4 => ElementOrder::Tet4,
            MeshElementOrderArg::Tet10 => ElementOrder::Tet10,
        },
        deterministic_seed: command.deterministic_seed,
        target_edge_length_m: runmat_types::physical::Length::new(command.target_size_m),
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(
            command.maximum_deviation_m,
        ),
        ..MeshingRequestSettings::default()
    };
    settings.limits.maximum_elements = command.maximum_elements;
    settings
}

fn import_options(command: &MeshCommand) -> ExactCadImportOptions {
    let mut options = ExactCadImportOptions {
        source_units: Some(match command.source_units {
            MeshSourceUnitsArg::Meter => LengthUnit::Meter,
            MeshSourceUnitsArg::Millimeter => LengthUnit::Millimeter,
            MeshSourceUnitsArg::Inch => LengthUnit::Inch,
        }),
        ..ExactCadImportOptions::default()
    };
    options.analysis.healing.shared_interfaces = command.shared_interfaces;
    options
}

fn print_result(
    command: &MeshCommand,
    output: &Path,
    evidence: &Path,
    result: &MeshOutputSummary,
) -> Result<()> {
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "solver_mesh_artifact_path": output,
                "meshing_evidence_artifact_path": evidence,
                "canonical_digest": hex_digest(result.canonical_digest.bytes()),
                "node_count": result.node_count,
                "element_count": result.element_count,
                "boundary_face_count": result.boundary_face_count,
                "resource_usage": result.resources,
                "stages": result.stages,
            }))?
        );
    } else {
        let styles = presentation::stdout();
        println!("{}", styles.success("Solver mesh generated"));
        println!(
            "  {} {}",
            styles.label("mesh:"),
            styles.path(output.display())
        );
        println!(
            "  {} {}",
            styles.label("evidence:"),
            styles.path(evidence.display())
        );
        println!(
            "  {} {} nodes, {} tetrahedra, {} boundary faces",
            styles.label("topology:"),
            result.node_count,
            result.element_count,
            result.boundary_face_count
        );
    }
    Ok(())
}

pub(crate) fn meshing_execution_provider() -> MeshingExecutionProvider {
    native_meshing_execution_provider()
}

fn validate_output_paths(output: &Path, evidence: &Path, force: bool) -> Result<()> {
    if output == evidence {
        anyhow::bail!("solver mesh and meshing evidence require distinct output paths");
    }
    if !force {
        for path in [output, evidence] {
            if path.exists() {
                anyhow::bail!(
                    "output {} already exists; use --force to replace it",
                    path.display()
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn hex_digest(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::validate_output_paths;

    #[test]
    fn output_paths_must_be_distinct_and_do_not_clobber_by_default() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("mesh.cbor");
        let evidence = directory.path().join("evidence.cbor");
        assert!(validate_output_paths(&output, &output, false).is_err());

        runmat_runtime::meshing::write_canonical_artifact_pair(
            &output,
            b"first mesh",
            &evidence,
            b"first evidence",
            false,
        )
        .unwrap();
        assert!(validate_output_paths(&output, &evidence, false).is_err());
        assert!(runmat_runtime::meshing::write_canonical_artifact_pair(
            &output,
            b"second mesh",
            &evidence,
            b"second evidence",
            false,
        )
        .is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"first mesh");
        assert_eq!(std::fs::read(&evidence).unwrap(), b"first evidence");
    }

    #[test]
    fn force_replaces_an_existing_output_atomically() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("mesh.cbor");
        let evidence = directory.path().join("evidence.cbor");
        runmat_runtime::meshing::write_canonical_artifact_pair(
            &output,
            b"first mesh",
            &evidence,
            b"first evidence",
            false,
        )
        .unwrap();
        runmat_runtime::meshing::write_canonical_artifact_pair(
            &output,
            b"second mesh",
            &evidence,
            b"second evidence",
            true,
        )
        .unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), b"second mesh");
        assert_eq!(std::fs::read(&evidence).unwrap(), b"second evidence");
    }
}

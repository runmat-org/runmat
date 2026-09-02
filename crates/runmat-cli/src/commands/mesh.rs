use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use runmat_execution_runner_native::{
    mesh_source_with_native_execution, native_meshing_execution_provider,
    NativeMeshingPartitionPolicy, NativeMeshingSourceRequest,
};
use runmat_geometry_core::UnitSystem;
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{
    CanonicalMeshingContract, ElementOrder, MeshingEvidence, MeshingRequestSettings,
    SolverMeshArtifact,
};
use runmat_meshing_execution::ExactMeshingDagRunResult;
use runmat_runtime::meshing::MeshingExecutionProvider;

use crate::cli::{MeshElementOrderArg, MeshSourceUnitsArg};
use crate::presentation;

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
    pub force: bool,
    pub json: bool,
}

pub fn execute(command: MeshCommand) -> Result<()> {
    let output = command
        .output
        .clone()
        .unwrap_or_else(|| command.source.with_extension("solver-mesh.cbor"));
    let evidence = command
        .evidence
        .clone()
        .unwrap_or_else(|| command.source.with_extension("meshing-evidence.cbor"));
    validate_output_paths(&output, &evidence, command.force)?;

    let mut settings = MeshingRequestSettings {
        element_order: match command.element_order {
            MeshElementOrderArg::Tet4 => ElementOrder::Tet4,
            MeshElementOrderArg::Tet10 => ElementOrder::Tet10,
        },
        deterministic_seed: command.deterministic_seed,
        target_edge_length_m: command.target_size_m,
        maximum_chordal_deviation_m: command.maximum_deviation_m,
        ..MeshingRequestSettings::default()
    };
    settings.resources.maximum_elements = command.maximum_elements;
    let generated = mesh_source_with_native_execution(NativeMeshingSourceRequest {
        source: command.source.clone(),
        import_options: ExactCadImportOptions {
            source_units: match command.source_units {
                MeshSourceUnitsArg::Meter => UnitSystem::Meter,
                MeshSourceUnitsArg::Millimeter => UnitSystem::Millimeter,
                MeshSourceUnitsArg::Inch => UnitSystem::Inch,
            },
            ..ExactCadImportOptions::default()
        },
        contact_definitions: Vec::new(),
        expected_source_sha256: None,
        settings,
        partition_policy: NativeMeshingPartitionPolicy::default(),
    })
    .map_err(anyhow::Error::new)?;
    let result = &generated.dag_result;

    let artifact_bytes = result
        .artifact
        .canonical_encode()
        .context("failed to encode solver mesh")?;
    let evidence_bytes = result
        .evidence
        .canonical_encode()
        .context("failed to encode meshing evidence")?;
    runmat_runtime::meshing::write_canonical_artifact_pair(
        &output,
        &artifact_bytes,
        &evidence,
        &evidence_bytes,
        command.force,
    )
    .map_err(anyhow::Error::msg)?;
    verify_persisted_outputs(&output, &evidence, &result.artifact, &result.evidence)?;

    print_result(&command, &output, &evidence, result)
}

fn print_result(
    command: &MeshCommand,
    output: &Path,
    evidence: &Path,
    result: &ExactMeshingDagRunResult,
) -> Result<()> {
    if command.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "solver_mesh_artifact_path": output,
                "meshing_evidence_artifact_path": evidence,
                "canonical_digest": hex_digest(result.artifact.canonical_digest.bytes()),
                "node_count": result.artifact.topology.nodes.len(),
                "element_count": result.artifact.topology.volume_elements.len(),
                "boundary_face_count": result.artifact.topology.boundary_faces.len(),
                "resource_usage": result.evidence.resources,
                "stages": result.evidence.stages,
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
            result.artifact.topology.nodes.len(),
            result.artifact.topology.volume_elements.len(),
            result.artifact.topology.boundary_faces.len()
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

fn verify_persisted_outputs(
    output: &Path,
    evidence_path: &Path,
    expected_artifact: &SolverMeshArtifact,
    expected_evidence: &MeshingEvidence,
) -> Result<()> {
    let artifact_bytes = std::fs::read(output).with_context(|| {
        format!(
            "failed to verify persisted solver mesh {}",
            output.display()
        )
    })?;
    let evidence_bytes = std::fs::read(evidence_path).with_context(|| {
        format!(
            "failed to verify persisted meshing evidence {}",
            evidence_path.display()
        )
    })?;
    let artifact = SolverMeshArtifact::canonical_decode(&artifact_bytes)
        .context("persisted solver mesh failed canonical decoding")?;
    let evidence = MeshingEvidence::canonical_decode(&evidence_bytes)
        .context("persisted meshing evidence failed canonical decoding")?;
    evidence
        .validate(&artifact)
        .context("persisted meshing outputs failed cross-validation")?;
    if &artifact != expected_artifact || &evidence != expected_evidence {
        anyhow::bail!("persisted meshing outputs differ from the completed canonical result");
    }
    Ok(())
}

fn hex_digest(bytes: &[u8; 32]) -> String {
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

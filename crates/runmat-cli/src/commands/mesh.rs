use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_runner_native::{
    mesh_exact_geometry, NativeExactMeshingJob, NativeExecutionConfig, NativeMeshingDomain,
    NativeMeshingExecutionPolicy,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{
    CacheAdmissionDecision, CanonicalMeshingContract, ElementOrder, MeshingEvidence,
    PlatformBuildIdentity, SolverMeshArtifact, StableDigest,
};
use runmat_meshing_execution::{
    resolve_meshing_request, MeshingRequestSettings, MeshingRunEvidenceContext,
};

use crate::cli::MeshElementOrderArg;
use crate::presentation;

pub struct MeshCommand {
    pub source: PathBuf,
    pub output: Option<PathBuf>,
    pub evidence: Option<PathBuf>,
    pub target_size_m: f64,
    pub maximum_deviation_m: f64,
    pub element_order: MeshElementOrderArg,
    pub material: String,
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

    let source_bytes = runmat_filesystem::read(&command.source)
        .with_context(|| format!("failed to read {}", command.source.display()))?;
    let import_options = ExactCadImportOptions::default();
    let tolerance = runmat_geometry_core::GeometryTolerancePolicy {
        source_tolerance_m: 0.0,
        absolute_floor_m: import_options.analysis.absolute_tolerance_floor_m,
        model_relative_term: import_options.analysis.model_relative_tolerance,
        requested_deviation_m: import_options.analysis.requested_deviation_m,
        maximum_healing_displacement_m: import_options.analysis.maximum_healing_displacement_m,
    };
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
    let request = resolve_meshing_request(tolerance, settings)
        .context("invalid canonical meshing request")?;
    let request_bytes = request
        .canonical_encode()
        .context("failed to encode canonical meshing request")?;
    let cohort = format!("native-{}-{}", std::env::consts::ARCH, std::env::consts::OS);
    let build_digest = StableDigest::from_bytes(
        *Digest::sha256(format!("runmat-meshing-build:{}", env!("CARGO_PKG_VERSION")).as_bytes())
            .bytes(),
    );
    let program_revision = ProgramRevision::new(
        Digest::sha256(&request_bytes),
        Digest::sha256(&source_bytes),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(env!("CARGO_PKG_VERSION").as_bytes()),
            Digest::sha256(b"runmat-meshing-catalog/v1"),
            "matlab",
        )?,
    )?;
    let source_name = command.source.to_string_lossy().into_owned();
    let result = mesh_exact_geometry(
        NativeExecutionConfig::for_current_executable()
            .context("failed to locate the RunMat meshing worker executable")?,
        NativeExactMeshingJob {
            source_name: &source_name,
            source_bytes: &source_bytes,
            import_options,
            request,
            domain: NativeMeshingDomain {
                default_material_id: Some(command.material),
                region_materials: Default::default(),
            },
            program_revision,
            capability_cohort: Some(cohort.clone()),
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
            inventory_limits: ObjectInventoryLimits::default(),
            evidence: MeshingRunEvidenceContext {
                platform: PlatformBuildIdentity {
                    capability_cohort: cohort,
                    target_triple: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
                    build_digest,
                    exact_kernel_abi: None,
                },
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
            execution: NativeMeshingExecutionPolicy::default(),
        },
    )
    .context("exact meshing failed")?;

    write_atomic(
        &output,
        &result
            .artifact
            .canonical_encode()
            .context("failed to encode solver mesh")?,
        command.force,
    )?;
    write_atomic(
        &evidence,
        &result
            .evidence
            .canonical_encode()
            .context("failed to encode meshing evidence")?,
        command.force,
    )?;
    verify_persisted_outputs(&output, &evidence, &result.artifact, &result.evidence)?;

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

fn write_atomic(path: &Path, bytes: &[u8], force: bool) -> Result<()> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to create temporary output in {}", parent.display()))?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    if force {
        temporary
            .persist(path)
            .map_err(|error| error.error)
            .with_context(|| format!("failed to publish {}", path.display()))?;
    } else {
        temporary
            .persist_noclobber(path)
            .map_err(|error| error.error)
            .with_context(|| {
                format!(
                    "failed to publish {}; use --force to replace an existing file",
                    path.display()
                )
            })?;
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
    use super::{validate_output_paths, write_atomic};

    #[test]
    fn output_paths_must_be_distinct_and_do_not_clobber_by_default() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("mesh.cbor");
        let evidence = directory.path().join("evidence.cbor");
        assert!(validate_output_paths(&output, &output, false).is_err());

        write_atomic(&output, b"first", false).unwrap();
        assert!(validate_output_paths(&output, &evidence, false).is_err());
        assert!(write_atomic(&output, b"second", false).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"first");
    }

    #[test]
    fn force_replaces_an_existing_output_atomically() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("mesh.cbor");
        write_atomic(&output, b"first", false).unwrap();
        write_atomic(&output, b"second", true).unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), b"second");
    }
}

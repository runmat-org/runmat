use std::path::PathBuf;

use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_meshing_core::ElementOrder;
use runmat_meshing_core::{CanonicalMeshingContract, MeshingEvidence, SolverMeshArtifact};
#[cfg(test)]
use runmat_value::IntegerStorage;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

mod inspection;
mod metadata;
mod object;
mod options;
#[cfg(test)]
use inspection::quality_value;
pub use metadata::*;
pub(crate) use object::execution_output_from_value;
use object::{evidence_object, mesh_object, payload_from_value, MeshPayload};
pub use object::{EVIDENCE_CLASS, MESH_CLASS};
#[cfg(feature = "plot-core")]
use options::parse_display_options;
use options::{parse_force_option, parse_settings, text};

const GENERATE_NAME: &str = "meshing.generate";
const LOAD_NAME: &str = "meshing.load";
const SAVE_NAME: &str = "meshing.save";
const NODES_NAME: &str = "meshing.nodes";
const ELEMENTS_NAME: &str = "meshing.elements";
const BOUNDARY_FACES_NAME: &str = "meshing.boundaryFaces";
const REGIONS_NAME: &str = "meshing.regions";
const QUALITY_NAME: &str = "meshing.quality";
const EVIDENCE_NAME: &str = "meshing.evidence";
const MESH_METHOD_NAME: &str = "__runmat_meshing_mesh";

#[runtime_builtin(
    name = "meshing.generate",
    category = "meshing",
    summary = "Generate a validated solver mesh from a geometry asset.",
    keywords = "meshing,generate,solver,tetrahedron,cad",
    descriptor(crate::builtins::meshing::GENERATE_DESCRIPTOR),
    integer_capabilities(crate::builtins::meshing::GENERATE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn generate_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (asset_value, options) = args
        .split_first()
        .ok_or_else(|| error(GENERATE_NAME, "expected a geometry.Asset"))?;
    let asset = crate::builtins::geometry::geometry_asset_from_value_with_builtin(
        asset_value,
        GENERATE_NAME,
    )?;
    let settings = parse_settings(options)?;
    let output = crate::meshing::execute_meshing(&crate::meshing::MeshingExecutionRequest {
        source_path: PathBuf::from(&asset.source.path),
        source_units: asset.units,
        expected_source_sha256: crate::meshing::parse_source_sha256(&asset.source.sha256)
            .map_err(|message| error(GENERATE_NAME, message))?,
        settings,
        contact_definitions: Vec::new(),
    })
    .map_err(|message| error(GENERATE_NAME, message))?;
    output
        .evidence
        .validate(&output.artifact)
        .map_err(|cause| error(GENERATE_NAME, cause.to_string()))?;
    mesh_object(MeshPayload::from(output))
}

#[runtime_builtin(
    name = "meshing.load",
    category = "meshing",
    summary = "Load and jointly validate solver-mesh and evidence artifacts.",
    keywords = "meshing,load,artifact,evidence",
    descriptor(crate::builtins::meshing::LOAD_DESCRIPTOR),
    integer_audit(crate::builtins::meshing::INTEGER_AUDIT),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn load_builtin(mesh_path: String, evidence_path: String) -> BuiltinResult<Value> {
    let mesh_bytes = runmat_filesystem::read(&mesh_path)
        .map_err(|cause| error(LOAD_NAME, format!("failed to read {mesh_path}: {cause}")))?;
    let evidence_bytes = runmat_filesystem::read(&evidence_path).map_err(|cause| {
        error(
            LOAD_NAME,
            format!("failed to read {evidence_path}: {cause}"),
        )
    })?;
    let artifact = SolverMeshArtifact::canonical_decode(&mesh_bytes)
        .map_err(|cause| error(LOAD_NAME, cause.to_string()))?;
    let evidence = MeshingEvidence::canonical_decode(&evidence_bytes)
        .map_err(|cause| error(LOAD_NAME, cause.to_string()))?;
    evidence
        .validate(&artifact)
        .map_err(|cause| error(LOAD_NAME, cause.to_string()))?;
    mesh_object(MeshPayload {
        artifact,
        evidence,
        region_aliases: Default::default(),
    })
}

#[runtime_builtin(
    name = "meshing.save",
    category = "meshing",
    summary = "Save a meshing object as canonical mesh and evidence artifacts.",
    keywords = "meshing,save,artifact,evidence",
    descriptor(crate::builtins::meshing::SAVE_DESCRIPTOR),
    integer_audit(crate::builtins::meshing::INTEGER_AUDIT),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn save_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    let [mesh, mesh_path, evidence_path, options @ ..] = args.as_slice() else {
        return Err(error(
            SAVE_NAME,
            "expected mesh, mesh path, evidence path, and optional Name, Value options",
        ));
    };
    let mesh_path = text(SAVE_NAME, mesh_path)?;
    let evidence_path = text(SAVE_NAME, evidence_path)?;
    if mesh_path == evidence_path {
        return Err(error(SAVE_NAME, "mesh and evidence paths must be distinct"));
    }
    let force = parse_force_option(options)?;
    let payload = payload_from_value(SAVE_NAME, mesh)?;
    let mesh_bytes = payload
        .artifact
        .canonical_encode()
        .map_err(|cause| error(SAVE_NAME, cause.to_string()))?;
    let evidence_bytes = payload
        .evidence
        .canonical_encode()
        .map_err(|cause| error(SAVE_NAME, cause.to_string()))?;
    crate::meshing::write_canonical_artifact_pair(
        std::path::Path::new(&mesh_path),
        &mesh_bytes,
        std::path::Path::new(&evidence_path),
        &evidence_bytes,
        force,
    )
    .map_err(|cause| error(SAVE_NAME, cause))?;
    Ok(true)
}

#[runtime_builtin(
    name = "meshing.nodes",
    category = "meshing",
    summary = "Return solver-mesh node ids and coordinates.",
    keywords = "meshing,nodes,coordinates",
    descriptor(crate::builtins::meshing::NODES_DESCRIPTOR),
    integer_audit(crate::builtins::meshing::INTEGER_AUDIT),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn nodes_builtin(mesh: Value) -> BuiltinResult<Value> {
    inspection::nodes(mesh)
}

#[runtime_builtin(
    name = "meshing.elements",
    category = "meshing",
    summary = "Return solver-mesh volume element connectivity.",
    keywords = "meshing,elements,tetrahedra,connectivity",
    descriptor(crate::builtins::meshing::ELEMENTS_DESCRIPTOR),
    integer_audit(crate::builtins::meshing::INTEGER_AUDIT),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn elements_builtin(mesh: Value) -> BuiltinResult<Value> {
    inspection::elements(mesh)
}

#[runtime_builtin(
    name = "meshing.boundaryFaces",
    category = "meshing",
    summary = "Return solver-mesh boundary face connectivity and provenance ids.",
    keywords = "meshing,boundary,faces,triangles",
    descriptor(crate::builtins::meshing::BOUNDARY_FACES_DESCRIPTOR),
    integer_audit(crate::builtins::meshing::INTEGER_AUDIT),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn boundary_faces_builtin(mesh: Value) -> BuiltinResult<Value> {
    inspection::boundary_faces(mesh)
}

#[runtime_builtin(
    name = "meshing.regions",
    category = "meshing",
    summary = "Return persistent physical-region inventories.",
    keywords = "meshing,regions,physical",
    descriptor(crate::builtins::meshing::REGIONS_DESCRIPTOR),
    integer_audit(crate::builtins::meshing::INTEGER_AUDIT),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn regions_builtin(mesh: Value) -> BuiltinResult<Value> {
    inspection::regions(mesh)
}

#[runtime_builtin(
    name = "meshing.quality",
    category = "meshing",
    summary = "Compute per-element solver-mesh quality statistics.",
    keywords = "meshing,quality,jacobian,aspect,volume",
    descriptor(crate::builtins::meshing::QUALITY_DESCRIPTOR),
    integer_audit(crate::builtins::meshing::INTEGER_AUDIT),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn quality_builtin(mesh: Value) -> BuiltinResult<Value> {
    inspection::quality(mesh)
}

#[runtime_builtin(
    name = "meshing.evidence",
    category = "meshing",
    summary = "Return typed validation and resource evidence for a solver mesh.",
    keywords = "meshing,evidence,stages,resources",
    descriptor(crate::builtins::meshing::EVIDENCE_DESCRIPTOR),
    integer_audit(crate::builtins::meshing::INTEGER_AUDIT),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn evidence_builtin(mesh: Value) -> BuiltinResult<Value> {
    evidence_object(payload_from_value(EVIDENCE_NAME, &mesh)?.evidence)
}

#[runtime_builtin(
    name = "__runmat_meshing_mesh",
    category = "meshing",
    summary = "Implement mesh display for meshing.Mesh objects.",
    keywords = "meshing,mesh,display",
    descriptor(crate::builtins::meshing::METHOD_DESCRIPTOR),
    integer_capabilities(crate::builtins::meshing::METHOD_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::meshing"
)]
pub async fn mesh_method_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    #[cfg(feature = "plot-core")]
    {
        let (mesh, options) = args
            .split_first()
            .ok_or_else(|| error("mesh", "expected a meshing.Mesh"))?;
        let payload = payload_from_value("mesh", mesh)?;
        let display = parse_display_options(options)?;
        let figure = crate::meshing::solver_mesh_figure(&payload.artifact, "Solver mesh", display)
            .map_err(|message| error("mesh", message))?;
        let handle = crate::builtins::plotting::import_runtime_figure(figure);
        Ok(Value::Num(f64::from(handle)))
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = args;
        Err(error(
            "mesh",
            "mesh display requires the plot-core runtime feature",
        ))
    }
}

fn error(builtin: &'static str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(format!("{builtin}: {}", message.into()))
        .with_builtin(builtin)
        .with_identifier("RunMat:meshing:InvalidInput")
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetrahedron_mesh_value() -> Value {
        let artifact =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let evidence =
            runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&artifact);
        mesh_object(MeshPayload {
            artifact,
            evidence,
            region_aliases: Default::default(),
        })
        .expect("fixture mesh object")
    }

    #[test]
    fn quality_returns_aligned_exact_ids_and_summary_statistics() {
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let Value::Struct(quality) = quality_value(&mesh).expect("fixture quality") else {
            panic!("expected quality struct");
        };
        let Value::Tensor(ids) = quality.fields.get("element_ids").expect("element ids") else {
            panic!("expected integer id tensor");
        };

        assert!(
            matches!(ids.integer_storage(), Some(IntegerStorage::U64(values)) if values.as_slice() == [1])
        );
        let Some(Value::Num(total_volume)) = quality.fields.get("total_corner_volume_m3") else {
            panic!("expected total volume");
        };
        assert!((*total_volume - 1.0 / 6.0).abs() < 1.0e-15);
        assert!(quality.fields.contains_key("p05_corner_scaled_jacobian"));
        assert!(quality.fields.contains_key("p95_corner_aspect_ratio"));
    }

    #[test]
    fn boundary_faces_preserve_topology_roles_adjacency_and_full_identity() {
        let Value::Struct(faces) =
            futures::executor::block_on(boundary_faces_builtin(tetrahedron_mesh_value()))
                .expect("boundary faces")
        else {
            panic!("expected boundary face struct");
        };

        let Value::Cell(roles) = faces.fields.get("roles").expect("roles") else {
            panic!("expected role cell");
        };
        assert_eq!(roles.shape, [4, 1]);
        assert!(roles
            .data
            .iter()
            .all(|role| role == &Value::String("exterior".to_string())));

        let Value::Cell(adjacency) = faces.fields.get("adjacent_element_ids").expect("adjacency")
        else {
            panic!("expected adjacency cell");
        };
        assert_eq!(adjacency.shape, [4, 1]);
        assert!(adjacency.data.iter().all(|ids| {
            matches!(
                ids,
                Value::Tensor(tensor)
                    if matches!(
                        tensor.integer_storage(),
                        Some(IntegerStorage::U64(values)) if values.as_slice() == [1]
                    )
            )
        }));

        let Value::Cell(provenance) = faces.fields.get("provenance").expect("provenance") else {
            panic!("expected provenance cell");
        };
        assert_eq!(provenance.shape, [4, 1]);
        let Value::Cell(first_face) = &provenance.data[0] else {
            panic!("expected per-face provenance cell");
        };
        let Value::Struct(identity) = &first_face.data[0] else {
            panic!("expected persistent identity struct");
        };
        assert_eq!(
            identity.fields.get("kind"),
            Some(&Value::String("face".into()))
        );
        assert_eq!(
            identity.fields.get("source_topology_id"),
            Some(&Value::String("face:0".into()))
        );
        assert!(identity.fields.contains_key("assembly_path"));
    }

    #[test]
    fn regions_preserve_persistent_identity_components() {
        let Value::Cell(regions) =
            futures::executor::block_on(regions_builtin(tetrahedron_mesh_value()))
                .expect("regions")
        else {
            panic!("expected region cell");
        };
        assert_eq!(regions.shape, [1, 1]);
        let Value::Struct(region) = &regions.data[0] else {
            panic!("expected region struct");
        };
        assert_eq!(
            region.fields.get("id"),
            Some(&Value::String("region".into()))
        );
        assert_eq!(
            region.fields.get("kind"),
            Some(&Value::String("region".into()))
        );
        assert!(region.fields.contains_key("assembly_path"));
        assert_eq!(
            region.fields.get("element_count"),
            Some(&Value::Int(runmat_value::IntValue::U64(1)))
        );
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn solver_mesh_display_accepts_region_and_section_options() {
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let options = crate::meshing::SolverMeshFigureOptions {
            show_edges: true,
            regions: Some(["region".to_string()].into_iter().collect()),
            section_plane: Some([1.0, 0.0, 0.0, -0.25]),
        };

        let figure =
            crate::meshing::solver_mesh_figure(&mesh, "section", options).expect("section figure");
        assert_eq!(figure.title.as_deref(), Some("section"));
        assert_eq!(figure.len(), 2);
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn standard_mesh_dispatches_solver_meshes_through_object_protocol() {
        let handle = futures::executor::block_on(crate::call_builtin_async(
            "mesh",
            &[tetrahedron_mesh_value()],
        ))
        .expect("standard mesh should dispatch meshing.Mesh");
        assert!(matches!(handle, Value::Num(value) if value > 0.0));
    }
}

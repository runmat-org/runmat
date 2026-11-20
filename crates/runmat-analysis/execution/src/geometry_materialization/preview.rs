//! Materialized study geometry for display, shared by the desktop and browser hosts.
//!
//! A host executes the geometry program, then both hosts derive the exact product, the
//! tessellated preview and the study's region overlays here, so every host draws the
//! same faces under the same region names.
use std::collections::BTreeSet;

use runmat_execution::value::ValuePayload;
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_geometry_core::GeometryAsset;
use runmat_geometry_io::{GeometryImportContext, GeometryImportOptions};

use super::{
    execute_geometry_materialization_program_request, import_geometry_materialization_output,
    preview_regions::add_materialized_regions, GeometryMaterializationLimits,
    GeometryMaterializationOutput,
};

/// The exact product of one study's geometry materialization.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedFeaGeometry {
    pub output: GeometryMaterializationOutput,
    pub topology: runmat_geometry_core::ExactBRepTopology,
    pub kernel_representation: Vec<u8>,
}

/// The display product a host returns: the tessellated asset with the study's regions,
/// and the size of the exact representation it was tessellated from.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MaterializedGeometryPreview {
    pub asset: GeometryAsset,
    pub kernel_representation_bytes: u64,
}

/// Reads the exact topology and kernel representation behind a materialization output and
/// checks that its display face ordinals cover exactly the topology's faces.
pub fn materialized_fea_geometry(
    store: &impl CacheImport,
    output: GeometryMaterializationOutput,
    access: ArtifactAccess,
    inventory: ObjectInventoryLimits,
) -> Result<MaterializedFeaGeometry, String> {
    let exact = runmat_meshing_execution::import_exact_geometry_input(
        store,
        output.document.clone(),
        &output.geometry_root,
        access,
        inventory,
    )
    .map_err(|error| error.to_string())?;
    let components = exact
        .into_geometry_objects()
        .into_components()
        .map_err(|error| error.to_string())?;
    let topology = components.topology;
    let mapped_faces = output
        .source_face_ids
        .values()
        .cloned()
        .collect::<BTreeSet<_>>();
    let topology_faces = topology
        .faces
        .iter()
        .map(|face| face.id.clone())
        .collect::<BTreeSet<_>>();
    if mapped_faces != topology_faces {
        return Err(
            "materialized FEA geometry face ordinals do not match its exact topology".into(),
        );
    }
    let kernel_representation = components.kernel_representation.ok_or_else(|| {
        "materialized FEA geometry has no exact kernel representation".to_string()
    })?;
    Ok(MaterializedFeaGeometry {
        output,
        topology,
        kernel_representation,
    })
}

/// Tessellates a materialized study geometry under its own identities, names each face and body
/// as the CAD author did, and adds the study's resolved regions as face overlays. `options.source_units` is taken from the materialized
/// document.
pub fn materialized_geometry_preview(
    materialized: &MaterializedFeaGeometry,
    display_name: &str,
    mut options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<MaterializedGeometryPreview, String> {
    options.source_units = Some(
        materialized
            .output
            .document
            .source
            .representation_units
            .unit(),
    );
    let imported = runmat_geometry_io::import_materialized_geometry_preview(
        display_name,
        &materialized.kernel_representation,
        options,
        context,
        &materialized.topology,
        &materialized.output.source_face_ids,
        &materialized.output.display_names,
    )
    .map_err(|error| error.to_string())?;
    let mut asset = imported.asset;
    add_materialized_regions(
        &mut asset,
        &materialized.output.regions,
        &materialized.topology,
        &materialized.output.source_face_ids,
    );
    asset.validate().map_err(str::to_string)?;
    Ok(MaterializedGeometryPreview {
        asset,
        kernel_representation_bytes: materialized.kernel_representation.len() as u64,
    })
}

/// Executes a prepared geometry program in one serial exact host and returns its display
/// preview. The browser geometry host runs this; the desktop materializes natively and calls
/// [`materialized_geometry_preview`] directly.
pub fn materialize_preview_serial<S: CacheImport + CacheExport>(
    program: &ProgramExecutionRequest,
    store: &mut S,
    platform: runmat_meshing_core::PlatformBuildIdentity,
    inventory: ObjectInventoryLimits,
    display_name: &str,
    options: GeometryImportOptions,
    context: &GeometryImportContext,
) -> Result<MaterializedGeometryPreview, String> {
    let descriptor = match program.arguments.first() {
        Some(ValuePayload::Object(value)) => value,
        _ => return Err("Geometry preview has no admitted descriptor.".into()),
    };
    let access = ArtifactAccess {
        authorization_scope: descriptor.authorization_scope.clone(),
        encryption_context: descriptor.encryption_context,
    };
    let response = execute_geometry_materialization_program_request(
        program,
        store,
        GeometryMaterializationLimits { inventory },
        platform,
    );
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = response else {
        return Err(format!(
            "Geometry preview failed to materialize: {response:?}"
        ));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err("Geometry preview returned an invalid root.".into());
    };
    let output =
        import_geometry_materialization_output(store, root, &access).map_err(|e| e.to_string())?;
    let materialized = materialized_fea_geometry(store, output, access, inventory)?;
    materialized_geometry_preview(&materialized, display_name, options, context)
}

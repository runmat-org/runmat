//! Cold geometry imports retain original independently checked component bytes.
use super::*;

pub fn import_exact_geometry_objects(
    source: &impl CacheImport,
    document: GeometryDocument,
    root: ExactGeometryObjectRoot,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<PreparedExactGeometryObjects> {
    import_exact_geometry_objects_controlled(source, document, root, limits, None)
}
pub(super) fn import_exact_geometry_objects_controlled(
    source: &impl CacheImport,
    document: GeometryDocument,
    root: ExactGeometryObjectRoot,
    limits: ObjectInventoryLimits,
    control: Option<&dyn runmat_geometry_core::GeometryCanonicalControl>,
) -> MeshingExecutionResult<PreparedExactGeometryObjects> {
    if let Some(control) = control {
        control
            .checkpoint()
            .map_err(|error| MeshingExecutionError::Invalid(error.to_string()))?;
    }
    document.validate()?;
    if document.primary_artifact().digest.bytes() != root.digest.bytes()
        || document.primary_artifact().encoded_length != root.encoded_length
    {
        return Err(MeshingExecutionError::Identity(
            "exact geometry root differs from the document artifact",
        ));
    }
    enforce_object_length("exact geometry manifest", root.encoded_length, limits)?;
    let manifest_bytes = read_exact(source, root.digest, root.encoded_length)?;
    let manifest = match control {
        Some(control) => {
            ExactGeometryManifest::canonical_decode_with_control(&manifest_bytes, control)?
        }
        None => ExactGeometryManifest::canonical_decode(&manifest_bytes)?,
    };
    let object_count = 3
        + usize::from(manifest.kernel_representation.is_some())
        + usize::from(manifest.healing_report.is_some());
    if object_count > limits.max_objects {
        return Err(ArtifactError::Limit("too many exact geometry objects".into()).into());
    }
    let mut total_bytes = root.encoded_length;
    let topology_bytes = read_component(
        source,
        "exact geometry topology",
        &manifest.topology,
        &mut total_bytes,
        limits,
    )?;
    let evaluator_bytes = read_component(
        source,
        "exact geometry evaluators",
        &manifest.evaluators,
        &mut total_bytes,
        limits,
    )?;
    let kernel_representation = manifest
        .kernel_representation
        .as_ref()
        .map(|reference| {
            read_component(
                source,
                "kernel representation",
                reference,
                &mut total_bytes,
                limits,
            )
        })
        .transpose()?;
    let healing_bytes = manifest
        .healing_report
        .as_ref()
        .map(|reference| {
            read_component(
                source,
                "geometry healing report",
                reference,
                &mut total_bytes,
                limits,
            )
        })
        .transpose()?;
    let GeometryModel::ExactBRep { model } = &document.model else {
        return Err(MeshingExecutionError::Invalid(
            "faceted geometry cannot import an exact closure".into(),
        ));
    };
    let bytes = runmat_geometry_core::ExactGeometryComponentBytes {
        topology: &topology_bytes,
        evaluators: &evaluator_bytes,
        healing: healing_bytes.as_deref(),
    };
    let runmat_geometry_core::DecodedExactGeometryComponents {
        topology,
        evaluators,
        healing_report,
    } = match control {
        Some(control) => runmat_geometry_core::decode_exact_geometry_components_with_control(
            bytes, model, control,
        )?,
        None => runmat_geometry_core::decode_exact_geometry_components(bytes, model)?,
    };
    // Raw decode and content checks have already validated each component.
    // Check their shared bindings, then retain the exact provider buffers.
    runmat_geometry_core::validate_exact_geometry_component_bindings(
        &document,
        &manifest,
        &topology,
        &evaluators,
        kernel_representation.as_deref(),
        healing_report.as_ref(),
    )?;
    admit_import_storage(control, import_slots::<LogicalObject>(object_count)?)?;
    let named = |prefix: &str, media: &str, bytes: Vec<u8>| {
        admit_import_storage(
            control,
            ObjectDescriptor::content_named_text_bytes(prefix, media)?,
        )?;
        geometry_object(prefix, media, bytes)
    };
    let mut objects = Vec::with_capacity(object_count);
    objects.push(named(
        "geometry/canonical/topology",
        EXACT_TOPOLOGY_MEDIA_TYPE,
        topology_bytes,
    )?);
    objects.push(named(
        "geometry/canonical/evaluators",
        EXACT_EVALUATOR_MEDIA_TYPE,
        evaluator_bytes,
    )?);
    if let Some(bytes) = kernel_representation {
        objects.push(named(
            "geometry/canonical/kernel",
            KERNEL_REPRESENTATION_MEDIA_TYPE,
            bytes,
        )?);
    }
    if let Some(bytes) = healing_bytes {
        objects.push(named(
            "geometry/canonical/healing",
            GEOMETRY_HEALING_MEDIA_TYPE,
            bytes,
        )?);
    }
    let manifest_object = named(
        "geometry/canonical/manifests",
        EXACT_BREP_MEDIA_TYPE,
        manifest_bytes,
    )?;
    admit_import_storage(
        control,
        (manifest_object.descriptor.logical_name.len()
            + manifest_object.descriptor.media_type.len()) as u64,
    )?;
    let descriptor = manifest_object.descriptor.clone();
    objects.push(manifest_object);
    validate_inventory(&objects, limits)?;
    let prepared = PreparedExactGeometryObjects {
        document,
        manifest,
        topology,
        evaluators,
        healing_report,
        root: descriptor,
        objects,
    };
    if prepared.root.digest != root.digest || prepared.root.encoded_length != root.encoded_length {
        return Err(MeshingExecutionError::Identity(
            "imported exact geometry manifest differs from requested root",
        ));
    }
    Ok(prepared)
}

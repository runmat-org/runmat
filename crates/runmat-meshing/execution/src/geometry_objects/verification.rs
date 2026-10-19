//! Raw mutable closures are verified against borrowed typed values and original
//! inventory bytes. Revalidation never clones topology, evaluators or kernel bytes.
use super::*;
pub(super) fn revalidate(
    value: &PreparedExactGeometryObjects,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<()> {
    validate_inventory(&value.objects, limits)?;
    let encoded = build_exact_geometry_closure(
        value.document.clone(),
        &value.topology,
        &value.evaluators,
        value.kernel_representation()?,
        value.healing_report.as_ref(),
    )?;
    if encoded.document != value.document || encoded.manifest != value.manifest {
        return Err(invalid());
    }
    let expected = 3
        + usize::from(value.manifest.kernel_representation.is_some())
        + usize::from(value.healing_report.is_some());
    if value.objects.len() != expected {
        return Err(invalid());
    }
    let mut index = 0;
    check(
        &value.objects[index],
        "geometry/canonical/topology",
        EXACT_TOPOLOGY_MEDIA_TYPE,
        &encoded.topology_bytes,
    )?;
    index += 1;
    check(
        &value.objects[index],
        "geometry/canonical/evaluators",
        EXACT_EVALUATOR_MEDIA_TYPE,
        &encoded.evaluator_bytes,
    )?;
    index += 1;
    if let Some(bytes) = value.kernel_representation()? {
        check(
            &value.objects[index],
            "geometry/canonical/kernel",
            KERNEL_REPRESENTATION_MEDIA_TYPE,
            bytes,
        )?;
        index += 1;
    }
    if let Some(bytes) = encoded.healing_bytes.as_deref() {
        check(
            &value.objects[index],
            "geometry/canonical/healing",
            GEOMETRY_HEALING_MEDIA_TYPE,
            bytes,
        )?;
        index += 1;
    }
    check(
        &value.objects[index],
        "geometry/canonical/manifests",
        EXACT_BREP_MEDIA_TYPE,
        &encoded.manifest_bytes,
    )?;
    if value.root != value.objects[index].descriptor {
        return Err(invalid());
    }
    Ok(())
}
fn check(
    object: &LogicalObject,
    prefix: &str,
    media: &str,
    expected_bytes: &[u8],
) -> MeshingExecutionResult<()> {
    crate::object_support::check_canonical_object(
        object,
        ObjectNamespace::InputValue,
        prefix,
        media,
        expected_bytes,
    )
}
fn invalid() -> MeshingExecutionError {
    MeshingExecutionError::Identity("prepared exact geometry is not its canonical object closure")
}

use std::path::Path;

use runmat_analysis_language::{
    GeometryInspection, GeometryInspectionCapability, GeometryInspectionRequest,
    SourceAvailability, SourceDiagnostic, SourceDigest,
};
use runmat_geometry_core::{Ap242Capability, Ap242CapabilityDisposition, Ap242ExchangeInventory};
use runmat_geometry_io::import::GeometryImportError;
use runmat_geometry_io::{
    detect_geometry_format, import_exact_cad, import_geometry, ExactCadImportOptions,
    GeometryFormat, GeometryImportContext, GeometryImportOptions, SourceUnitsAuthority,
};
use sha2::{Digest, Sha256};

use super::host::{diagnostic, source_digest};

pub(super) fn inspect_geometry(
    path: &Path,
    request: &GeometryInspectionRequest,
) -> Result<GeometryInspection, SourceDiagnostic> {
    let bytes = runmat_filesystem::read(path).map_err(|error| {
        diagnostic(
            "fea.host.geometry_unavailable",
            format!("cannot read geometry {}: {error}", path.display()),
            SourceAvailability::Unavailable,
        )
    })?;
    inspect_geometry_bytes(&bytes, request)
}

pub(super) fn inspect_geometry_bytes(
    bytes: &[u8],
    request: &GeometryInspectionRequest,
) -> Result<GeometryInspection, SourceDiagnostic> {
    let logical_path = request.logical_path.as_str();
    let content_digest = source_digest(bytes);
    if request
        .expected_content_digest
        .is_some_and(|expected| expected != content_digest)
    {
        return Err(diagnostic(
            "fea.host.geometry_digest_mismatch",
            format!("geometry `{logical_path}` differs from its expected digest"),
            SourceAvailability::Available,
        ));
    }
    // A check runs on every edit and every agent check; the CAD import and
    // exact closure dominate it. Identical bytes, units and path inspect
    // identically, so reuse the result until the file changes.
    let key = (
        logical_path.to_string(),
        content_digest,
        format!("{:?}/{}", request.source_units, request.shared_interfaces),
    );
    if let Some(cached) = inspection_cache::get(&key) {
        return Ok(cached);
    }
    let inspection = inspect_uncached(bytes, request, logical_path, content_digest)?;
    inspection_cache::insert(key, inspection.clone());
    Ok(inspection)
}

fn inspect_uncached(
    bytes: &[u8],
    request: &GeometryInspectionRequest,
    logical_path: &str,
    content_digest: SourceDigest,
) -> Result<GeometryInspection, SourceDiagnostic> {
    // Authored `units` are authoritative: they reinterpret a file declaring a different single
    // length unit (the declaration stays in the evidence as provenance). Without authored units
    // the file's own single declared unit applies.
    let inspected = runmat_geometry_io::source_unit_evidence(logical_path, bytes);
    let unit_evidence = match request.source_units {
        Some(authored) => inspected
            .apply_authored(authored, "authored geometry units")
            .unwrap_or_else(|unresolved| *unresolved),
        None => inspected,
    };
    let units = match request.source_units {
        Some(authored) => authored,
        None => unit_evidence
            .uniform_unit()
            .ok_or_else(|| undeclared_units(bytes, logical_path))?,
    };
    let imported = import_geometry(
        logical_path,
        bytes,
        GeometryImportOptions {
            units,
            source_units: request.source_units,
            ..GeometryImportOptions::default()
        },
    )
    .map_err(|error| {
        diagnostic(
            "fea.host.geometry_inspection_failed",
            format!("cannot inspect geometry `{logical_path}`: {error}"),
            SourceAvailability::Available,
        )
    })?;
    let exchange = imported.asset.source_geometry.exchange.clone();
    let format = detect_geometry_format(logical_path, bytes);
    let exact = matches!(
        format,
        GeometryFormat::Step | GeometryFormat::Iges | GeometryFormat::Brep
    )
    .then(|| {
        import_exact_cad(
            logical_path,
            bytes,
            format,
            &{
                let mut options = ExactCadImportOptions {
                    source_units: Some(units),
                    source_units_authority: SourceUnitsAuthority::Authored,
                    ..ExactCadImportOptions::default()
                };
                options.analysis.healing.shared_interfaces = request.shared_interfaces;
                options
            },
            &GeometryImportContext::new(),
        )
    })
    .transpose();
    let mut authored_names = std::collections::BTreeMap::<String, Vec<String>>::new();
    let (topology_digest, exact_topology) = match exact {
        Ok(Some(exact)) => {
            for (id, names) in &exact.entity_names {
                let role = match id.kind {
                    runmat_geometry_core::PersistentEntityKind::Vertex => "vertex",
                    runmat_geometry_core::PersistentEntityKind::Edge => "edge",
                    runmat_geometry_core::PersistentEntityKind::Face => "face",
                    runmat_geometry_core::PersistentEntityKind::Solid => "solid",
                    _ => continue,
                };
                authored_names
                    .entry(role.to_string())
                    .or_default()
                    .extend(names.iter().cloned());
            }
            // One entry per named entity: a name carried by two entities appears twice, so a
            // check can count what `select name` resolves to.
            for names in authored_names.values_mut() {
                names.sort();
            }
            let closure = exact.build_closure().map_err(|error| {
                diagnostic(
                    "fea.host.geometry_inspection_failed",
                    format!("cannot close exact geometry `{logical_path}`: {error}"),
                    SourceAvailability::Available,
                )
            })?;
            (
                SourceDigest(*closure.manifest.topology.digest.bytes()),
                true,
            )
        }
        Ok(None) => (preview_topology_digest(&imported.asset)?, false),
        // Browser hosts can inspect/tessellate imported CAD without a native
        // exact-BREP backend. Keep those facts available for setup and preview;
        // do not advertise exact topology or persistent IDs from this fallback.
        Err(GeometryImportError::BackendUnavailable(_)) => {
            (preview_topology_digest(&imported.asset)?, false)
        }
        Err(_error)
            if imported.asset.source_geometry.cad_evaluators.is_empty()
                && imported.asset.region_entity_mappings.is_empty() =>
        {
            (preview_topology_digest(&imported.asset)?, false)
        }
        Err(error) => {
            return Err(diagnostic(
                "fea.host.geometry_inspection_failed",
                format!("cannot inspect exact geometry `{logical_path}`: {error}"),
                SourceAvailability::Available,
            ));
        }
    };
    Ok(GeometryInspection {
        preview_extent_m: preview_extent_m(&imported.asset),
        content_digest,
        topology_digest,
        capabilities: capabilities(&imported.asset, exchange.as_ref(), exact_topology),
        exchange,
        resolved_units: Some(units),
        authored_names,
        unit_evidence,
    })
}

/// An imported geometry without authored `units` uses the single length unit its STEP or IGES
/// file declares. A BREP must state its unit; an exchange file without one declared length unit
/// cannot be read in any authored unit either and must be re-exported with one.
fn undeclared_units(bytes: &[u8], logical_path: &str) -> SourceDiagnostic {
    let exchange = matches!(
        detect_geometry_format(logical_path, bytes),
        GeometryFormat::Step | GeometryFormat::Iges
    );
    diagnostic(
        "fea.host.geometry_units_undeclared",
        if exchange {
            format!(
                "geometry `{logical_path}` does not declare a single length unit (none, or several different ones); re-export it with one declared unit"
            )
        } else {
            format!(
                "geometry `{logical_path}` declares no length unit; add `units = mm`, `m`, or `in`"
            )
        },
        SourceAvailability::Available,
    )
}

fn preview_extent_m(asset: &runmat_geometry_core::GeometryAsset) -> Option<[f64; 3]> {
    let scale = asset.coordinate_units?.meters_per_unit();
    let mut points = asset
        .surface_meshes
        .iter()
        .flat_map(|mesh| mesh.vertices.iter());
    let first = *points.next()?;
    let (mut min, mut max) = (first, first);
    for point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    let extent = std::array::from_fn(|axis| (max[axis] - min[axis]) * scale);
    extent
        .iter()
        .all(|x| x.is_finite() && *x > 0.0)
        .then_some(extent)
}

fn preview_topology_digest(
    asset: &runmat_geometry_core::GeometryAsset,
) -> Result<SourceDigest, SourceDiagnostic> {
    let bytes = serde_json::to_vec(&(
        &asset.source_geometry,
        &asset.regions,
        &asset.region_entity_mappings,
    ))
    .map_err(|error| {
        diagnostic(
            "fea.host.geometry_fingerprint_failed",
            format!("cannot fingerprint inspected geometry topology: {error}"),
            SourceAvailability::Available,
        )
    })?;
    Ok(SourceDigest(Sha256::digest(bytes).into()))
}

fn capabilities(
    asset: &runmat_geometry_core::GeometryAsset,
    exchange: Option<&Ap242ExchangeInventory>,
    exact_topology: bool,
) -> Vec<GeometryInspectionCapability> {
    let mut capabilities = Vec::new();
    if exact_topology
        || !asset.source_geometry.cad_evaluators.is_empty()
        || asset
            .regions
            .iter()
            .any(|region| region.cad_ownership.is_some())
    {
        capabilities.push(GeometryInspectionCapability::Topology);
    }
    if exact_topology {
        capabilities.push(GeometryInspectionCapability::PersistentIds);
    }
    let Some(exchange) = exchange else {
        return capabilities;
    };
    if (supported(exchange, Ap242Capability::PersistentIds) || !exchange.persistent_ids.is_empty())
        && !capabilities.contains(&GeometryInspectionCapability::PersistentIds)
    {
        capabilities.push(GeometryInspectionCapability::PersistentIds);
    }
    if supported(exchange, Ap242Capability::Parameters) && !exchange.parameters.is_empty() {
        capabilities.push(GeometryInspectionCapability::Parameters);
    }
    if supported(exchange, Ap242Capability::EngineeringProperties)
        && !exchange.engineering_properties.is_empty()
    {
        capabilities.push(GeometryInspectionCapability::EngineeringProperties);
    }
    if supported(exchange, Ap242Capability::Materials) && !exchange.materials.is_empty() {
        capabilities.push(GeometryInspectionCapability::MaterialCandidates);
    }
    if exchange.capability(Ap242Capability::Regeneration) == Ap242CapabilityDisposition::Regenerable
    {
        capabilities.push(GeometryInspectionCapability::Regeneration);
    }
    capabilities
}

fn supported(exchange: &Ap242ExchangeInventory, capability: Ap242Capability) -> bool {
    matches!(
        exchange.capability(capability),
        Ap242CapabilityDisposition::ReadOnly
            | Ap242CapabilityDisposition::ReadWrite
            | Ap242CapabilityDisposition::Regenerable
    )
}

mod inspection_cache {
    use std::collections::VecDeque;
    use std::sync::{Mutex, OnceLock};

    use runmat_analysis_language::{GeometryInspection, SourceDigest};

    pub(super) type Key = (String, SourceDigest, String);
    const CAPACITY: usize = 32;

    fn entries() -> &'static Mutex<VecDeque<(Key, GeometryInspection)>> {
        static ENTRIES: OnceLock<Mutex<VecDeque<(Key, GeometryInspection)>>> = OnceLock::new();
        ENTRIES.get_or_init(|| Mutex::new(VecDeque::new()))
    }

    pub(super) fn get(key: &Key) -> Option<GeometryInspection> {
        let mut entries = entries().lock().unwrap_or_else(|error| error.into_inner());
        let index = entries.iter().position(|(candidate, _)| candidate == key)?;
        let entry = entries.remove(index)?;
        let inspection = entry.1.clone();
        entries.push_back(entry);
        Some(inspection)
    }

    pub(super) fn insert(key: Key, inspection: GeometryInspection) {
        let mut entries = entries().lock().unwrap_or_else(|error| error.into_inner());
        entries.retain(|(candidate, _)| candidate != &key);
        if entries.len() == CAPACITY {
            entries.pop_front();
        }
        entries.push_back((key, inspection));
    }
}

#[cfg(test)]
mod preflight_extent_tests {
    #[test]
    fn preview_extents_use_resolved_physical_scale() {
        let imported = runmat_geometry_io::import_geometry(
            "tetra.obj",
            b"v 0 0 0\nv 10 0 0\nv 0 20 0\nv 0 0 30\nf 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n",
            runmat_geometry_io::GeometryImportOptions {
                units: runmat_geometry_core::LengthUnit::Millimeter,
                source_units: Some(runmat_geometry_core::LengthUnit::Millimeter),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            super::preview_extent_m(&imported.asset),
            Some([0.01, 0.02, 0.03])
        );
        let mut empty = imported.asset;
        empty.surface_meshes.clear();
        assert_eq!(super::preview_extent_m(&empty), None);
    }
}

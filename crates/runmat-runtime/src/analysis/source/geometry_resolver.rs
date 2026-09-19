use std::path::Path;

use runmat_analysis_language::{
    GeometryInspection, GeometryInspectionCapability, GeometryInspectionRequest,
    SourceAvailability, SourceDiagnostic, SourceDigest,
};
use runmat_geometry_core::{Ap242Capability, Ap242CapabilityDisposition, Ap242ExchangeInventory};
use runmat_geometry_io::{
    detect_geometry_format, import_exact_cad, import_geometry, ExactCadImportOptions,
    GeometryFormat, GeometryImportContext, GeometryImportOptions,
};
use sha2::{Digest, Sha256};

use super::host::{diagnostic, digest};

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
    let content_digest = digest(&bytes);
    if request
        .expected_content_digest
        .is_some_and(|expected| expected != content_digest)
    {
        return Err(diagnostic(
            "fea.host.geometry_digest_mismatch",
            format!(
                "geometry {} differs from its expected digest",
                path.display()
            ),
            SourceAvailability::Available,
        ));
    }
    let logical_path = request.logical_path.as_str();
    let imported = import_geometry(
        logical_path,
        &bytes,
        GeometryImportOptions {
            units: request.source_units,
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
    let format = detect_geometry_format(logical_path, &bytes);
    let exact = matches!(
        format,
        GeometryFormat::Step | GeometryFormat::Iges | GeometryFormat::Brep
    )
    .then(|| {
        import_exact_cad(
            logical_path,
            &bytes,
            format,
            &ExactCadImportOptions {
                source_units: request.source_units,
                ..ExactCadImportOptions::default()
            },
            &GeometryImportContext::new(),
        )
    })
    .transpose();
    let (topology_digest, exact_topology) = match exact {
        Ok(Some(exact)) => {
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
        content_digest,
        topology_digest,
        capabilities: capabilities(&imported.asset, exchange.as_ref(), exact_topology),
        exchange,
    })
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
    if supported(exchange, Ap242Capability::PersistentIds) || !exchange.persistent_ids.is_empty() {
        if !capabilities.contains(&GeometryInspectionCapability::PersistentIds) {
            capabilities.push(GeometryInspectionCapability::PersistentIds);
        }
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

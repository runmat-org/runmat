use runmat_geometry_core::{GeometryDigest, GeometryRevisionIdentity, UnitSystem};
use runmat_geometry_ops::{
    GeometryFeatureCapabilities, GeometryFeatureKind, GeometryFeatureProvider,
    GeometryFeatureProviderOutput, ResolvedGeometryFeatureOperation,
};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    exact::{ExactCadImportOptions, ImportedExactCad},
    import::{GeometryImportContext, GeometryImportError},
    GeometryFormat,
};

use super::{ffi, native};

#[derive(Debug, Error)]
pub enum OcctExactFeatureError {
    #[error("invalid OCCT feature request: {0}")]
    InvalidRequest(String),
    #[error("OCCT feature kernel failed: {0}")]
    Kernel(String),
    #[error(transparent)]
    Import(#[from] GeometryImportError),
    #[error("exact geometry contract failed: {0}")]
    Contract(String),
}

#[derive(Debug, Clone)]
pub struct OcctExactFeatureProvider {
    options: ExactCadImportOptions,
    context: GeometryImportContext,
    maximum_features: u64,
    maximum_instances_per_feature: u64,
    maximum_boolean_operands: u64,
}

impl OcctExactFeatureProvider {
    pub fn new(options: ExactCadImportOptions, context: GeometryImportContext) -> Self {
        Self {
            options,
            context,
            maximum_features: 10_000,
            maximum_instances_per_feature: 10_000,
            maximum_boolean_operands: 1_000,
        }
    }

    pub fn with_plan_bounds(
        mut self,
        maximum_features: u64,
        maximum_instances_per_feature: u64,
        maximum_boolean_operands: u64,
    ) -> Result<Self, OcctExactFeatureError> {
        if maximum_features == 0
            || maximum_instances_per_feature == 0
            || maximum_boolean_operands < 2
        {
            return Err(OcctExactFeatureError::InvalidRequest(
                "feature and instance bounds must be positive and boolean bound must be at least two"
                    .into(),
            ));
        }
        self.maximum_features = maximum_features;
        self.maximum_instances_per_feature = maximum_instances_per_feature;
        self.maximum_boolean_operands = maximum_boolean_operands;
        Ok(self)
    }
}

impl GeometryFeatureProvider for OcctExactFeatureProvider {
    type Geometry = ImportedExactCad;
    type Error = OcctExactFeatureError;

    fn capabilities(&self) -> GeometryFeatureCapabilities {
        GeometryFeatureCapabilities {
            supported: [
                GeometryFeatureKind::Box,
                GeometryFeatureKind::Cylinder,
                GeometryFeatureKind::Sphere,
                GeometryFeatureKind::Transform,
                GeometryFeatureKind::Instance,
                GeometryFeatureKind::Union,
                GeometryFeatureKind::Subtract,
                GeometryFeatureKind::Intersect,
            ]
            .into_iter()
            .collect(),
            maximum_features: self.maximum_features,
            maximum_instances_per_feature: self.maximum_instances_per_feature,
            maximum_boolean_operands: self.maximum_boolean_operands,
        }
    }

    fn execute(
        &self,
        feature_id: &str,
        operation: &ResolvedGeometryFeatureOperation,
        dependencies: &[(&str, &Self::Geometry)],
        _source: Option<&Self::Geometry>,
    ) -> Result<GeometryFeatureProviderOutput<Self::Geometry>, Self::Error> {
        self.context.check_cancelled()?;
        if !self.capabilities().supports(operation.kind()) {
            return Err(OcctExactFeatureError::InvalidRequest(format!(
                "operation {:?} is outside the qualified OCCT feature profile",
                operation.kind()
            )));
        }
        validate_dependencies(operation, dependencies)?;
        let encoded = encode_operation(operation)?;
        let operands = dependencies
            .iter()
            .map(|(_, geometry)| ffi::bridge::OcctFeatureOperand {
                representation: geometry.representation.clone(),
            })
            .collect();
        let cancel_token = ffi::OcctCancelTokenRegistration::new(self.context.cancellation_flag());
        let payload = ffi::bridge::execute_exact_feature(
            encoded.kind,
            &encoded.scalars,
            &encoded.matrices,
            operands,
            native::ffi_exact_import_options(&self.options, cancel_token.id(), 1.0),
        )
        .map_err(|error| OcctExactFeatureError::Kernel(error.to_string()))?;

        let mut options = self.options.clone();
        options.source_units = UnitSystem::Meter;
        options.analysis.revision = derived_revision(dependencies)?;
        let geometry = crate::import_exact_cad(
            &format!("{feature_id}.brep"),
            &payload.representation,
            GeometryFormat::Brep,
            &options,
            &self.context,
        )?;
        let history =
            super::feature_history::topology_history(dependencies, &geometry.topology, &payload)?;
        Ok(GeometryFeatureProviderOutput { geometry, history })
    }
}

fn validate_dependencies(
    operation: &ResolvedGeometryFeatureOperation,
    dependencies: &[(&str, &ImportedExactCad)],
) -> Result<(), OcctExactFeatureError> {
    use ResolvedGeometryFeatureOperation as Operation;
    let expected = match operation {
        Operation::Box { .. } | Operation::Cylinder { .. } | Operation::Sphere { .. } => Vec::new(),
        Operation::Transform { source, .. } | Operation::Instance { source, .. } => {
            vec![source.0.as_str()]
        }
        Operation::Union { operands } | Operation::Intersect { operands } => {
            operands.iter().map(|operand| operand.0.as_str()).collect()
        }
        Operation::Subtract { base, tools } => std::iter::once(base.0.as_str())
            .chain(tools.iter().map(|tool| tool.0.as_str()))
            .collect(),
        _ => {
            return Err(OcctExactFeatureError::InvalidRequest(format!(
                "operation {:?} is outside the qualified OCCT feature profile",
                operation.kind()
            )))
        }
    };
    let actual = dependencies
        .iter()
        .map(|(dependency, _)| *dependency)
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(OcctExactFeatureError::InvalidRequest(format!(
            "operation {:?} expected dependencies {expected:?} but received {actual:?}",
            operation.kind()
        )));
    }
    Ok(())
}

struct EncodedOperation {
    kind: ffi::bridge::OcctFeatureKind,
    scalars: Vec<f64>,
    matrices: Vec<f64>,
}

fn encode_operation(
    operation: &ResolvedGeometryFeatureOperation,
) -> Result<EncodedOperation, OcctExactFeatureError> {
    use ffi::bridge::OcctFeatureKind as Kind;
    use ResolvedGeometryFeatureOperation as Operation;
    let (kind, scalars, matrices) = match operation {
        Operation::Box {
            width_m,
            depth_m,
            height_m,
        } => (Kind::Box, vec![*width_m, *depth_m, *height_m], Vec::new()),
        Operation::Cylinder {
            radius_m,
            height_m,
            axis,
            origin_m,
        } => (
            Kind::Cylinder,
            std::iter::once(*radius_m)
                .chain(std::iter::once(*height_m))
                .chain(axis.iter().copied())
                .chain(origin_m.iter().copied())
                .collect(),
            Vec::new(),
        ),
        Operation::Sphere { radius_m, center_m } => (
            Kind::Sphere,
            std::iter::once(*radius_m)
                .chain(center_m.iter().copied())
                .collect(),
            Vec::new(),
        ),
        Operation::Transform { matrix, .. } => (Kind::Transform, Vec::new(), matrix.to_vec()),
        Operation::Instance { transforms, .. } => (
            Kind::Instance,
            Vec::new(),
            transforms.iter().flatten().copied().collect(),
        ),
        Operation::Union { .. } => (Kind::Union, Vec::new(), Vec::new()),
        Operation::Subtract { .. } => (Kind::Subtract, Vec::new(), Vec::new()),
        Operation::Intersect { .. } => (Kind::Intersect, Vec::new(), Vec::new()),
        unsupported => {
            return Err(OcctExactFeatureError::InvalidRequest(format!(
                "operation {:?} is outside the bounded OCCT feature profile",
                unsupported.kind()
            )))
        }
    };
    Ok(EncodedOperation {
        kind,
        scalars,
        matrices,
    })
}

fn derived_revision(
    dependencies: &[(&str, &ImportedExactCad)],
) -> Result<GeometryRevisionIdentity, OcctExactFeatureError> {
    let revision = dependencies
        .iter()
        .map(|(_, geometry)| geometry.analysis_options().revision.revision)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| OcctExactFeatureError::Contract("geometry revision overflow".into()))?;
    let persistent_mapping_version = dependencies
        .iter()
        .map(|(_, geometry)| {
            geometry
                .analysis_options()
                .revision
                .persistent_mapping_version
        })
        .max()
        .unwrap_or(1);
    let parent_document_digest = if dependencies.is_empty() {
        None
    } else {
        let mut lineage = Sha256::new();
        lineage.update(b"runmat.geometry.feature-parent-set\0");
        for (dependency_id, geometry) in dependencies {
            let digest = geometry
                .build_closure()
                .map(|closure| closure.manifest.topology.digest)
                .map_err(|error| OcctExactFeatureError::Contract(error.to_string()))?;
            lineage.update((dependency_id.len() as u64).to_be_bytes());
            lineage.update(dependency_id.as_bytes());
            lineage.update(digest.bytes());
        }
        Some(GeometryDigest::from_bytes(lineage.finalize().into()))
    };
    Ok(GeometryRevisionIdentity {
        revision,
        persistent_mapping_version,
        parent_document_digest,
    })
}

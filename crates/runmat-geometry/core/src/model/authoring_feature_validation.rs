use runmat_canonical_codec::CanonicalLimits;
use std::collections::BTreeSet;

use super::authoring::{
    token, GeneratedGeometryProduct, GeneratedGeometryRegionOutput, GeometryFeatureGraph,
    GeometryFeatureOperation, GeometryRegionSelector,
};
use super::{GeometryContractError, GeometryScalarExpression};

impl GeometryFeatureGraph {
    pub(crate) fn validate(
        &self,
        source_geometry: Option<&str>,
        parameters: &BTreeSet<&str>,
    ) -> Result<(), GeometryContractError> {
        if self.features.is_empty() {
            return Err(GeometryContractError::invalid(
                "feature graph",
                "requires at least one feature",
            ));
        }
        let mut prior = BTreeSet::new();
        for feature in &self.features {
            token("feature id", &feature.feature_id)?;
            if !prior.insert(feature.feature_id.as_str()) {
                return Err(GeometryContractError::invalid(
                    "feature graph",
                    "feature ids must be unique",
                ));
            }
            feature
                .operation
                .validate(&prior, source_geometry, parameters)?;
        }
        if !prior.contains(self.result.0.as_str()) {
            return Err(GeometryContractError::invalid(
                "feature graph result",
                "must reference a declared feature",
            ));
        }
        Ok(())
    }
}

impl GeneratedGeometryRegionOutput {
    const CODEC_DOMAIN: &'static str = "generated-geometry-region-output";
    const CODEC_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024, 256, 64, 16);

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GeometryContractError> {
        self.selector.validate()?;
        super::canonical::encode(Self::CODEC_DOMAIN, self, Self::CODEC_LIMITS)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, GeometryContractError> {
        let output = super::canonical::decode(Self::CODEC_DOMAIN, bytes, Self::CODEC_LIMITS)?;
        Self::canonical_bytes(&output)?;
        Ok(output)
    }
}

impl GeneratedGeometryProduct {
    const CODEC_DOMAIN: &'static str = "generated-geometry-product";
    const CODEC_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 100_000, 100_000, 64);

    pub fn validate(&self) -> Result<(), GeometryContractError> {
        let parameters = BTreeSet::new();
        self.graph.validate(None, &parameters)?;
        for (name, output) in &self.region_outputs {
            token("generated geometry region output", name)?;
            match &output.selector {
                GeometryRegionSelector::Result => {}
                GeometryRegionSelector::FeatureResult(feature)
                | GeometryRegionSelector::BoundaryIntroducedBy(feature)
                    if self
                        .graph
                        .features
                        .iter()
                        .any(|candidate| candidate.feature_id == feature.0) => {}
                GeometryRegionSelector::FeatureResult(_)
                | GeometryRegionSelector::BoundaryIntroducedBy(_) => {
                    return Err(GeometryContractError::invalid(
                        "generated geometry region output",
                        "must reference a feature in the returned graph",
                    ));
                }
                GeometryRegionSelector::Name(_)
                | GeometryRegionSelector::Tag(_)
                | GeometryRegionSelector::PersistentId(_)
                | GeometryRegionSelector::GeneratedOutput(_) => {
                    return Err(GeometryContractError::invalid(
                        "generated geometry region output",
                        "must select the result or a typed feature output",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GeometryContractError> {
        self.validate()?;
        super::canonical::encode(Self::CODEC_DOMAIN, self, Self::CODEC_LIMITS)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, GeometryContractError> {
        let product = super::canonical::decode(Self::CODEC_DOMAIN, bytes, Self::CODEC_LIMITS)?;
        Self::validate(&product)?;
        Ok(product)
    }
}

impl GeometryFeatureOperation {
    pub fn canonical_digest(&self) -> Result<super::GeometryDigest, GeometryContractError> {
        let bytes = super::canonical::encode(
            "geometry-feature-operation",
            self,
            CanonicalLimits::new(1024 * 1024, 100_000, 100_000, 64),
        )?;
        super::canonical::digest(&bytes)
    }

    fn validate(
        &self,
        prior: &BTreeSet<&str>,
        source: Option<&str>,
        parameters: &BTreeSet<&str>,
    ) -> Result<(), GeometryContractError> {
        self.validate_references(prior, source)?;
        match self {
            Self::Box {
                width_m,
                depth_m,
                height_m,
            } => positive("box dimensions", [width_m, depth_m, height_m], parameters)?,
            Self::Cylinder {
                radius_m,
                height_m,
                axis,
                origin_m,
            } => {
                positive("cylinder dimensions", [radius_m, height_m], parameters)?;
                direction("cylinder axis", axis, parameters)?;
                expressions(origin_m, parameters)?;
            }
            Self::Sphere { radius_m, center_m } => {
                positive("sphere radius", [radius_m], parameters)?;
                expressions(center_m, parameters)?;
            }
            Self::Transform { matrix, .. } => {
                matrix_components("transform matrix", matrix, parameters)?
            }
            Self::Instance { transforms, .. } => {
                if transforms.is_empty() {
                    return Err(GeometryContractError::invalid(
                        "instances",
                        "requires at least one transform",
                    ));
                }
                for transform in transforms {
                    matrix_components("instance transform", transform, parameters)?;
                }
            }
            Self::Extrude {
                distance_m,
                direction: axis,
                ..
            } => {
                positive("extrusion distance", [distance_m], parameters)?;
                direction("extrusion direction", axis, parameters)?;
            }
            Self::Revolve {
                axis,
                origin_m,
                angle_rad,
                ..
            } => {
                direction("revolution axis", axis, parameters)?;
                expressions(origin_m, parameters)?;
                positive("revolution angle", [angle_rad], parameters)?;
            }
            Self::Union { operands } | Self::Intersect { operands } if operands.len() < 2 => {
                return Err(GeometryContractError::invalid(
                    "boolean operands",
                    "requires at least two operands",
                ));
            }
            Self::Subtract { tools, .. } if tools.is_empty() => {
                return Err(GeometryContractError::invalid(
                    "subtraction tools",
                    "requires at least one tool",
                ));
            }
            Self::Hole {
                radius_m,
                depth_m,
                axis,
                origin_m,
                ..
            } => {
                positive("hole dimensions", [radius_m, depth_m], parameters)?;
                direction("hole axis", axis, parameters)?;
                expressions(origin_m, parameters)?;
            }
            Self::Pocket {
                depth_m,
                direction: axis,
                ..
            } => {
                positive("pocket depth", [depth_m], parameters)?;
                direction("pocket direction", axis, parameters)?;
            }
            Self::Fillet {
                edges, radius_m, ..
            } => {
                edges.validate_edge_selection(prior, source)?;
                positive("fillet radius", [radius_m], parameters)?;
            }
            Self::Chamfer {
                edges, distance_m, ..
            } => {
                edges.validate_edge_selection(prior, source)?;
                positive("chamfer distance", [distance_m], parameters)?;
            }
            Self::RegenerateSource { parameters: values } => {
                for value in values.values() {
                    value.validate(parameters)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn validate_references(
        &self,
        prior: &BTreeSet<&str>,
        source: Option<&str>,
    ) -> Result<(), GeometryContractError> {
        let mut references = Vec::new();
        match self {
            Self::Transform { source, .. } | Self::Instance { source, .. } => {
                references.push(source)
            }
            Self::Extrude { profile, .. } | Self::Revolve { profile, .. } => {
                references.push(profile)
            }
            Self::Union { operands } | Self::Intersect { operands } => references.extend(operands),
            Self::Subtract { base, tools } => {
                references.push(base);
                references.extend(tools);
            }
            Self::Hole { target, .. } => references.push(target),
            Self::Pocket {
                target, profile, ..
            } => {
                references.push(target);
                references.push(profile);
            }
            Self::Fillet { target, .. } | Self::Chamfer { target, .. } => references.push(target),
            Self::RegenerateSource { .. } if source.is_none() => {
                return Err(GeometryContractError::invalid(
                    "regenerate source",
                    "is valid only in derived geometry",
                ));
            }
            _ => {}
        }
        if references.iter().any(|reference| {
            !(prior.contains(reference.0.as_str())
                || source.is_some() && reference.0 == "source.result")
        }) {
            return Err(GeometryContractError::invalid(
                "feature dependency",
                "must reference an earlier feature",
            ));
        }
        Ok(())
    }
}

impl super::GeometryFeatureRegionSelection {
    fn validate_edge_selection(
        &self,
        prior: &BTreeSet<&str>,
        source: Option<&str>,
    ) -> Result<(), GeometryContractError> {
        if self.role != super::GeometryRegionRole::Edge {
            return Err(GeometryContractError::invalid(
                "feature edge selection",
                "must have edge topology",
            ));
        }
        self.selector.validate()?;
        match &self.selector {
            GeometryRegionSelector::FeatureResult(feature)
            | GeometryRegionSelector::BoundaryIntroducedBy(feature)
                if prior.contains(feature.0.as_str())
                    || source.is_some() && feature.0 == "source.result" => {}
            GeometryRegionSelector::FeatureResult(_)
            | GeometryRegionSelector::BoundaryIntroducedBy(_) => {
                return Err(GeometryContractError::invalid(
                    "feature edge selection",
                    "must reference an earlier feature",
                ));
            }
            GeometryRegionSelector::Result | GeometryRegionSelector::PersistentId(_) => {}
            GeometryRegionSelector::Name(_)
            | GeometryRegionSelector::Tag(_)
            | GeometryRegionSelector::GeneratedOutput(_) => {
                return Err(GeometryContractError::invalid(
                    "feature edge selection",
                    "must use result, persistent-id, or typed feature lineage",
                ));
            }
        }
        Ok(())
    }
}

fn expressions(
    values: &[GeometryScalarExpression],
    parameters: &BTreeSet<&str>,
) -> Result<(), GeometryContractError> {
    for value in values {
        value.validate(parameters)?;
    }
    Ok(())
}

fn positive<'a>(
    field: &str,
    values: impl IntoIterator<Item = &'a GeometryScalarExpression>,
    parameters: &BTreeSet<&str>,
) -> Result<(), GeometryContractError> {
    for value in values {
        value.validate_positive_literal(field, parameters)?;
    }
    Ok(())
}

fn matrix_components(
    field: &str,
    values: &[GeometryScalarExpression],
    parameters: &BTreeSet<&str>,
) -> Result<(), GeometryContractError> {
    if values.len() != 16 {
        return Err(GeometryContractError::invalid(
            field,
            "requires exactly 16 components",
        ));
    }
    expressions(values, parameters)
}

fn direction(
    field: &str,
    value: &[GeometryScalarExpression; 3],
    parameters: &BTreeSet<&str>,
) -> Result<(), GeometryContractError> {
    expressions(value, parameters)?;
    let literals = value
        .iter()
        .map(|component| match component {
            GeometryScalarExpression::Literal { value } => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    if literals.is_some_and(|values| values.iter().all(|value| *value == 0.0)) {
        return Err(GeometryContractError::invalid(
            field,
            "literal direction must be nonzero",
        ));
    }
    Ok(())
}

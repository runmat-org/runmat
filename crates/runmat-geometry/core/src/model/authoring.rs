use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::{GeometryContractError, GeometryDigest, GeometrySourceFormat, UnitSystem};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryDefinition {
    pub geometry_id: String,
    pub source: GeometryDefinitionSource,
    #[serde(default)]
    pub regions: Vec<GeometryRegionDefinition>,
    #[serde(default)]
    pub cad_parameters: Vec<CadParameterDefinition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GeometryDefinitionSource {
    Imported(ImportedGeometryDefinition),
    Constructed(GeometryFeatureGraph),
    Derived(DerivedGeometryDefinition),
    Generated(GeneratedGeometryDefinition),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedGeometryDefinition {
    pub logical_path: String,
    pub format: GeometrySourceFormat,
    pub units: UnitSystem,
    pub content_digest: GeometryDigest,
    pub topology_digest: GeometryDigest,
    pub capabilities: GeometryAuthoringCapabilities,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryAuthoringCapabilities {
    pub topology: bool,
    pub persistent_ids: bool,
    pub parameters: bool,
    pub engineering_properties: bool,
    pub material_candidates: bool,
    pub regeneration: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedGeometryDefinition {
    pub source_geometry_id: String,
    pub graph: GeometryFeatureGraph,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedGeometryDefinition {
    pub module_path: String,
    pub function: String,
    pub source_digest: GeometryDigest,
    pub arguments: BTreeMap<String, GeometryAuthoringValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryFeatureGraph {
    pub features: Vec<GeometryFeature>,
    pub result: GeometryFeatureRef,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryFeature {
    pub feature_id: String,
    pub operation: GeometryFeatureOperation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GeometryFeatureOperation {
    Box {
        width_m: f64,
        depth_m: f64,
        height_m: f64,
    },
    Cylinder {
        radius_m: f64,
        height_m: f64,
        axis: [f64; 3],
        origin_m: [f64; 3],
    },
    Sphere {
        radius_m: f64,
        center_m: [f64; 3],
    },
    Transform {
        source: GeometryFeatureRef,
        matrix: [f64; 16],
    },
    Instance {
        source: GeometryFeatureRef,
        transforms: Vec<[f64; 16]>,
    },
    Extrude {
        profile: GeometryFeatureRef,
        distance_m: f64,
        direction: [f64; 3],
    },
    Revolve {
        profile: GeometryFeatureRef,
        axis: [f64; 3],
        origin_m: [f64; 3],
        angle_rad: f64,
    },
    Union {
        operands: Vec<GeometryFeatureRef>,
    },
    Subtract {
        base: GeometryFeatureRef,
        tools: Vec<GeometryFeatureRef>,
    },
    Intersect {
        operands: Vec<GeometryFeatureRef>,
    },
    Hole {
        target: GeometryFeatureRef,
        radius_m: f64,
        depth_m: f64,
        axis: [f64; 3],
        origin_m: [f64; 3],
    },
    Pocket {
        target: GeometryFeatureRef,
        profile: GeometryFeatureRef,
        depth_m: f64,
    },
    Fillet {
        target: GeometryFeatureRef,
        edges: GeometryRegionRef,
        radius_m: f64,
    },
    Chamfer {
        target: GeometryFeatureRef,
        edges: GeometryRegionRef,
        distance_m: f64,
    },
    RegenerateSource {
        parameters: BTreeMap<String, GeometryAuthoringValue>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GeometryFeatureRef(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum GeometryAuthoringValue {
    Scalar(f64),
    Vector([f64; 3]),
    Text(String),
    Boolean(bool),
    Feature(GeometryFeatureRef),
    Region(GeometryRegionRef),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRegionRef {
    pub geometry_id: String,
    pub region_id: String,
    pub role: GeometryRegionRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRegionDefinition {
    pub region_id: String,
    pub role: GeometryRegionRole,
    pub selector: GeometryRegionSelector,
    pub cardinality: GeometryRegionCardinality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryRegionRole {
    Vertex,
    Edge,
    Face,
    Solid,
    Cell,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum GeometryRegionSelector {
    Name(String),
    Tag(String),
    PersistentId(String),
    Result,
    FeatureResult(GeometryFeatureRef),
    BoundaryIntroducedBy(GeometryFeatureRef),
    GeneratedOutput(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryRegionCardinality {
    ExactlyOne,
    OneOrMore,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CadParameterDefinition {
    pub parameter_id: String,
    pub physical_type: CadParameterPhysicalType,
    pub selector: CadParameterSelector,
    pub access: CadParameterAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadParameterPhysicalType {
    Dimensionless,
    Length,
    Angle,
    Area,
    Volume,
    Mass,
    Temperature,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum CadParameterSelector {
    Name(String),
    PersistentId(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadParameterAccess {
    Readable,
    Writable,
}

impl GeometryDefinition {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        token("geometry id", &self.geometry_id)?;
        match &self.source {
            GeometryDefinitionSource::Imported(source) => source.validate()?,
            GeometryDefinitionSource::Constructed(graph) => graph.validate(None)?,
            GeometryDefinitionSource::Derived(source) => {
                token("derived source geometry id", &source.source_geometry_id)?;
                graph_source_must_differ(&self.geometry_id, &source.source_geometry_id)?;
                source.graph.validate(Some(&source.source_geometry_id))?;
            }
            GeometryDefinitionSource::Generated(source) => source.validate()?,
        }
        let mut regions = std::collections::BTreeSet::new();
        for region in &self.regions {
            token("region id", &region.region_id)?;
            if !regions.insert(&region.region_id) {
                return Err(GeometryContractError::invalid(
                    "regions",
                    "region ids must be unique",
                ));
            }
            region.selector.validate()?;
        }
        let mut parameters = std::collections::BTreeSet::new();
        for parameter in &self.cad_parameters {
            token("CAD parameter id", &parameter.parameter_id)?;
            if !parameters.insert(&parameter.parameter_id) {
                return Err(GeometryContractError::invalid(
                    "cad parameters",
                    "parameter ids must be unique",
                ));
            }
        }
        Ok(())
    }
}

impl ImportedGeometryDefinition {
    fn validate(&self) -> Result<(), GeometryContractError> {
        token("geometry import path", &self.logical_path)?;
        if !self.format.is_exact() {
            return Err(GeometryContractError::invalid(
                "geometry import format",
                "analysis geometry must be exact",
            ));
        }
        if self.units == UnitSystem::Unspecified {
            return Err(GeometryContractError::invalid(
                "geometry import units",
                "units must be explicit",
            ));
        }
        self.content_digest
            .validate_nonzero("geometry content digest")?;
        self.topology_digest
            .validate_nonzero("geometry topology digest")?;
        if !self.capabilities.topology || !self.capabilities.persistent_ids {
            return Err(GeometryContractError::invalid(
                "geometry capabilities",
                "topology and persistent ids are required",
            ));
        }
        Ok(())
    }
}

impl GeneratedGeometryDefinition {
    fn validate(&self) -> Result<(), GeometryContractError> {
        token("generator module path", &self.module_path)?;
        token("generator function", &self.function)?;
        self.source_digest
            .validate_nonzero("generator source digest")
    }
}

impl GeometryFeatureGraph {
    pub(crate) fn validate(
        &self,
        source_geometry: Option<&str>,
    ) -> Result<(), GeometryContractError> {
        if self.features.is_empty() {
            return Err(GeometryContractError::invalid(
                "feature graph",
                "requires at least one feature",
            ));
        }
        let mut prior = std::collections::BTreeSet::new();
        for feature in &self.features {
            token("feature id", &feature.feature_id)?;
            if !prior.insert(feature.feature_id.as_str()) {
                return Err(GeometryContractError::invalid(
                    "feature graph",
                    "feature ids must be unique",
                ));
            }
            feature.operation.validate(&prior, source_geometry)?;
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

impl GeometryFeatureOperation {
    fn validate(
        &self,
        prior: &std::collections::BTreeSet<&str>,
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
                ))
            }
            _ => {}
        }
        if references
            .iter()
            .any(|reference| !prior.contains(reference.0.as_str()))
        {
            return Err(GeometryContractError::invalid(
                "feature dependency",
                "must reference an earlier feature",
            ));
        }
        match self {
            Self::Box {
                width_m,
                depth_m,
                height_m,
            } => positive_values("box dimensions", &[*width_m, *depth_m, *height_m])?,
            Self::Cylinder {
                radius_m,
                height_m,
                axis,
                origin_m,
            } => {
                positive_values("cylinder dimensions", &[*radius_m, *height_m])?;
                direction("cylinder axis", axis)?;
                finite_values("cylinder origin", origin_m)?;
            }
            Self::Sphere { radius_m, center_m } => {
                positive_values("sphere radius", &[*radius_m])?;
                finite_values("sphere center", center_m)?;
            }
            Self::Transform { matrix, .. } => finite_values("transform matrix", matrix)?,
            Self::Instance { transforms, .. } => {
                if transforms.is_empty() {
                    return Err(GeometryContractError::invalid(
                        "instances",
                        "requires at least one transform",
                    ));
                }
                for transform in transforms {
                    finite_values("instance transform", transform)?;
                }
            }
            Self::Extrude {
                distance_m,
                direction: axis,
                ..
            } => {
                positive_values("extrusion distance", &[*distance_m])?;
                direction("extrusion direction", axis)?;
            }
            Self::Revolve {
                axis,
                origin_m,
                angle_rad,
                ..
            } => {
                direction("revolution axis", axis)?;
                finite_values("revolution origin", origin_m)?;
                positive_values("revolution angle", &[*angle_rad])?;
            }
            Self::Union { operands } | Self::Intersect { operands } if operands.len() < 2 => {
                return Err(GeometryContractError::invalid(
                    "boolean operands",
                    "requires at least two operands",
                ))
            }
            Self::Subtract { tools, .. } if tools.is_empty() => {
                return Err(GeometryContractError::invalid(
                    "subtraction tools",
                    "requires at least one tool",
                ))
            }
            Self::Hole {
                radius_m,
                depth_m,
                axis,
                origin_m,
                ..
            } => {
                positive_values("hole dimensions", &[*radius_m, *depth_m])?;
                direction("hole axis", axis)?;
                finite_values("hole origin", origin_m)?;
            }
            Self::Pocket { depth_m, .. } => positive_values("pocket depth", &[*depth_m])?,
            Self::Fillet { radius_m, .. } => positive_values("fillet radius", &[*radius_m])?,
            Self::Chamfer { distance_m, .. } => {
                positive_values("chamfer distance", &[*distance_m])?
            }
            _ => {}
        }
        Ok(())
    }
}

impl GeometryRegionSelector {
    fn validate(&self) -> Result<(), GeometryContractError> {
        match self {
            Self::Name(value)
            | Self::Tag(value)
            | Self::PersistentId(value)
            | Self::GeneratedOutput(value) => token("region selector", value),
            Self::FeatureResult(reference) | Self::BoundaryIntroducedBy(reference) => {
                token("region feature", &reference.0)
            }
            Self::Result => Ok(()),
        }
    }
}

fn graph_source_must_differ(id: &str, source: &str) -> Result<(), GeometryContractError> {
    if id == source {
        Err(GeometryContractError::invalid(
            "derived geometry",
            "cannot derive from itself",
        ))
    } else {
        Ok(())
    }
}

fn token(field: &str, value: &str) -> Result<(), GeometryContractError> {
    if value.trim().is_empty() {
        Err(GeometryContractError::invalid(field, "must not be empty"))
    } else {
        Ok(())
    }
}

fn finite_values(field: &str, values: &[f64]) -> Result<(), GeometryContractError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(GeometryContractError::invalid(
            field,
            "values must be finite",
        ))
    }
}

fn positive_values(field: &str, values: &[f64]) -> Result<(), GeometryContractError> {
    if values.iter().all(|value| value.is_finite() && *value > 0.0) {
        Ok(())
    } else {
        Err(GeometryContractError::invalid(
            field,
            "values must be finite and positive",
        ))
    }
}

fn direction(field: &str, value: &[f64; 3]) -> Result<(), GeometryContractError> {
    finite_values(field, value)?;
    if value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        > 0.0
    {
        Ok(())
    } else {
        Err(GeometryContractError::invalid(
            field,
            "direction must be nonzero",
        ))
    }
}

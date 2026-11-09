use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::{
    Ap242ExchangeInventory, Ap242Parameter, Ap242ParameterAccess, GeometryAuthoringValue,
    GeometryContractError, GeometryDigest, GeometryScalarExpression, GeometrySourceFormat,
    LengthUnit, QuantityKind,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryDefinition {
    pub module_path: String,
    pub geometry_id: String,
    #[serde(default)]
    pub parameters: Vec<GeometryParameterDefinition>,
    pub source: GeometryDefinitionSource,
    #[serde(default)]
    pub regions: Vec<GeometryRegionDefinition>,
    #[serde(default)]
    pub cad_parameters: Vec<CadParameterDefinition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GeometryDefinitionSource {
    Imported(Box<ImportedGeometryDefinition>),
    Constructed(GeometryFeatureGraph),
    Derived(DerivedGeometryDefinition),
    Generated(GeneratedGeometryDefinition),
    /// A 2D board cross-section, meshed and solved by the transmission-line solver; no
    /// geometry kernel materializes it.
    CrossSection(super::CrossSectionDefinition),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedGeometryDefinition {
    /// Derived inspection evidence for advisory planning, not authored geometry
    /// or part of the exact-kernel materialization wire contract.
    #[serde(skip)]
    pub preview_extent_m: Option<[f64; 3]>,
    pub logical_path: String,
    pub format: GeometrySourceFormat,
    pub authored_units: Option<LengthUnit>,
    #[serde(default)]
    pub unit_evidence: runmat_types::physical::LengthUnitEvidence,
    pub content_digest: GeometryDigest,
    pub topology_digest: GeometryDigest,
    pub capabilities: GeometryAuthoringCapabilities,
    #[serde(default)]
    pub exchange: Option<Ap242ExchangeInventory>,
    /// Touching bodies join so each pair of coincident faces becomes one shared interface face
    /// (`interfaces = shared` in `.fea`); otherwise every part keeps its own faces.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub shared_interfaces: bool,
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
    pub source: GeometryDefinitionRef,
    pub graph: GeometryFeatureGraph,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryDefinitionRef {
    pub module_path: String,
    pub geometry_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedGeometryDefinition {
    pub module_path: String,
    pub function: String,
    pub source_digest: GeometryDigest,
    pub arguments: Vec<GeneratedGeometryArgument>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedGeometryArgument {
    pub parameter: String,
    pub value: GeometryAuthoringValue,
}

/// Provider-neutral result returned by an admitted RunMat geometry generator.
///
/// The generator describes exact construction intent; it never transports a
/// kernel handle, tessellation, or provider-specific geometry object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedGeometryProduct {
    pub graph: GeometryFeatureGraph,
    #[serde(default)]
    pub region_outputs: BTreeMap<String, GeneratedGeometryRegionOutput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedGeometryRegionOutput {
    pub role: GeometryRegionRole,
    pub selector: GeometryRegionSelector,
    pub cardinality: GeometryRegionCardinality,
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
#[serde(deny_unknown_fields)]
pub struct GeometryParameterDefinition {
    pub parameter_id: String,
    pub parameter_type: GeometryParameterType,
    pub required: bool,
    pub default: Option<GeometryAuthoringValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryParameterType {
    Boolean,
    Scalar,
    Length,
    Angle,
    Vector3,
    /// Numeric vectors carry the declared component descriptor; directions remain Vector3.
    QuantityVector3(runmat_types::physical::DynamicDimension),
    Text,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GeometryFeatureOperation {
    Box {
        width_m: GeometryScalarExpression,
        depth_m: GeometryScalarExpression,
        height_m: GeometryScalarExpression,
    },
    Cylinder {
        radius_m: GeometryScalarExpression,
        height_m: GeometryScalarExpression,
        axis: [GeometryScalarExpression; 3],
        origin_m: [GeometryScalarExpression; 3],
    },
    Sphere {
        radius_m: GeometryScalarExpression,
        center_m: [GeometryScalarExpression; 3],
    },
    Transform {
        source: GeometryFeatureRef,
        matrix: Vec<GeometryScalarExpression>,
    },
    Instance {
        source: GeometryFeatureRef,
        transforms: Vec<Vec<GeometryScalarExpression>>,
    },
    Extrude {
        profile: GeometryFeatureRef,
        distance_m: GeometryScalarExpression,
        direction: [GeometryScalarExpression; 3],
    },
    Revolve {
        profile: GeometryFeatureRef,
        axis: [GeometryScalarExpression; 3],
        origin_m: [GeometryScalarExpression; 3],
        angle_rad: GeometryScalarExpression,
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
        radius_m: GeometryScalarExpression,
        depth_m: GeometryScalarExpression,
        axis: [GeometryScalarExpression; 3],
        origin_m: [GeometryScalarExpression; 3],
    },
    Pocket {
        target: GeometryFeatureRef,
        profile: GeometryFeatureRef,
        depth_m: GeometryScalarExpression,
        direction: [GeometryScalarExpression; 3],
    },
    Fillet {
        target: GeometryFeatureRef,
        edges: GeometryFeatureRegionSelection,
        radius_m: GeometryScalarExpression,
    },
    Chamfer {
        target: GeometryFeatureRef,
        edges: GeometryFeatureRegionSelection,
        distance_m: GeometryScalarExpression,
    },
    RegenerateSource {
        parameters: BTreeMap<String, GeometryAuthoringValue>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GeometryFeatureRef(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRegionRef {
    pub module_path: String,
    pub geometry_id: String,
    pub region_id: String,
    pub role: GeometryRegionRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryFeatureRegionSelection {
    pub role: GeometryRegionRole,
    pub selector: GeometryRegionSelector,
    pub cardinality: GeometryRegionCardinality,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRegionDefinition {
    pub region_id: String,
    pub role: GeometryRegionRole,
    pub selector: GeometryRegionSelector,
    pub cardinality: GeometryRegionCardinality,
}

/// One authored geometry region after exact topology and cardinality have been resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedGeometryRegion {
    pub role: GeometryRegionRole,
    pub entity_ids: Vec<super::PersistentEntityId>,
}

/// Complete named-region inventory for one materialized geometry definition.
pub type ResolvedGeometryRegions = BTreeMap<String, ResolvedGeometryRegion>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CadParameterDefinition {
    pub parameter_id: String,
    pub physical_type: CadParameterPhysicalType,
    pub selector: CadParameterSelector,
    pub access: CadParameterAccess,
    pub source: Ap242Parameter,
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
        token("geometry module path", &self.module_path)?;
        token("geometry id", &self.geometry_id)?;
        let mut parameters = std::collections::BTreeSet::new();
        for parameter in &self.parameters {
            token("geometry parameter id", &parameter.parameter_id)?;
            if parameters.contains(parameter.parameter_id.as_str()) {
                return Err(GeometryContractError::invalid(
                    "geometry parameters",
                    "parameter ids must be unique",
                ));
            }
            if parameter.required == parameter.default.is_some() {
                return Err(GeometryContractError::invalid(
                    "geometry parameter",
                    "required parameters cannot have defaults and optional parameters require defaults",
                ));
            }
            if let Some(default) = &parameter.default {
                default.validate(&parameters)?;
            }
            parameters.insert(parameter.parameter_id.as_str());
        }
        match &self.source {
            GeometryDefinitionSource::Imported(source) => source.validate()?,
            GeometryDefinitionSource::Constructed(graph) => graph.validate(None, &parameters)?,
            GeometryDefinitionSource::Derived(source) => {
                source.source.validate()?;
                graph_source_must_differ(self, &source.source)?;
                source
                    .graph
                    .validate(Some(&source.source.geometry_id), &parameters)?;
            }
            GeometryDefinitionSource::Generated(source) => source.validate(&parameters)?,
            GeometryDefinitionSource::CrossSection(section) => section.validate()?,
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
            region.selector.validate_against_source(&self.source)?;
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
            parameter.validate()?;
        }
        Ok(())
    }
}

impl CadParameterDefinition {
    fn validate(&self) -> Result<(), GeometryContractError> {
        self.source.validate()?;
        let selected = match &self.selector {
            CadParameterSelector::Name(name) => self.source.name == *name,
            CadParameterSelector::PersistentId(id) => {
                self.source.persistent_id.as_deref() == Some(id.as_str())
            }
        };
        if !selected {
            return Err(GeometryContractError::invalid(
                "CAD parameter selector",
                "resolved source parameter does not match the authored selector",
            ));
        }
        let dimension = match self.physical_type {
            CadParameterPhysicalType::Dimensionless => QuantityKind::Dimensionless,
            CadParameterPhysicalType::Length => QuantityKind::Length,
            CadParameterPhysicalType::Angle => QuantityKind::Angle,
            CadParameterPhysicalType::Area => QuantityKind::Area,
            CadParameterPhysicalType::Volume => QuantityKind::Volume,
            CadParameterPhysicalType::Mass => QuantityKind::Mass,
            CadParameterPhysicalType::Temperature => QuantityKind::Temperature,
        };
        if self.source.value.kind() != dimension {
            return Err(GeometryContractError::invalid(
                "CAD parameter type",
                "authored physical type does not match the normalized source measure",
            ));
        }
        if self.access == CadParameterAccess::Writable
            && self.source.access != Ap242ParameterAccess::Regenerable
        {
            return Err(GeometryContractError::invalid(
                "CAD parameter access",
                "writable access requires a qualified regenerative source parameter",
            ));
        }
        Ok(())
    }
}

impl ImportedGeometryDefinition {
    fn validate(&self) -> Result<(), GeometryContractError> {
        if self
            .preview_extent_m
            .is_some_and(|extent| extent.iter().any(|x| !x.is_finite() || *x <= 0.0))
        {
            return Err(GeometryContractError::invalid(
                "geometry preview extent",
                "expected finite positive metres",
            ));
        }
        token("geometry import path", &self.logical_path)?;
        if !self.format.is_exact() {
            return Err(GeometryContractError::invalid(
                "geometry import format",
                "analysis geometry must be exact",
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
        if let Some(exchange) = &self.exchange {
            if self.format != GeometrySourceFormat::Step {
                return Err(GeometryContractError::invalid(
                    "geometry exchange inventory",
                    "AP242 exchange facts require a STEP source",
                ));
            }
            exchange.validate()?;
            use super::{Ap242Capability, Ap242CapabilityDisposition};
            let available = |capability| {
                !matches!(
                    exchange.capability(capability),
                    Ap242CapabilityDisposition::Absent
                        | Ap242CapabilityDisposition::PresentUnsupported
                )
            };
            if self.capabilities.parameters != available(Ap242Capability::Parameters)
                || self.capabilities.engineering_properties
                    != available(Ap242Capability::EngineeringProperties)
                || self.capabilities.material_candidates != available(Ap242Capability::Materials)
                || self.capabilities.regeneration != available(Ap242Capability::Regeneration)
            {
                return Err(GeometryContractError::invalid(
                    "geometry exchange capabilities",
                    "declared booleans must match the normalized AP242 inventory",
                ));
            }
        } else if self.capabilities.parameters
            || self.capabilities.engineering_properties
            || self.capabilities.material_candidates
            || self.capabilities.regeneration
        {
            return Err(GeometryContractError::invalid(
                "geometry exchange capabilities",
                "rich CAD capabilities require normalized exchange evidence",
            ));
        }
        Ok(())
    }
}

impl GeneratedGeometryDefinition {
    fn validate(
        &self,
        parameters: &std::collections::BTreeSet<&str>,
    ) -> Result<(), GeometryContractError> {
        token("generator module path", &self.module_path)?;
        token("generator function", &self.function)?;
        self.source_digest
            .validate_nonzero("generator source digest")?;
        let mut names = std::collections::BTreeSet::new();
        for argument in &self.arguments {
            token("generator parameter", &argument.parameter)?;
            if !names.insert(argument.parameter.as_str()) {
                return Err(GeometryContractError::invalid(
                    "generator arguments",
                    "parameter names must be unique",
                ));
            }
            argument.value.validate(parameters)?;
        }
        Ok(())
    }
}

impl GeometryRegionSelector {
    pub(crate) fn validate(&self) -> Result<(), GeometryContractError> {
        match self {
            Self::Name(value) | Self::Tag(value) | Self::GeneratedOutput(value) => {
                token("region selector", value)
            }
            Self::PersistentId(value) => {
                super::PersistentEntityId::validate_analysis_selection_id(value)
            }
            Self::FeatureResult(reference) | Self::BoundaryIntroducedBy(reference) => {
                token("region feature", &reference.0)
            }
            Self::Result => Ok(()),
        }
    }

    fn validate_against_source(
        &self,
        source: &GeometryDefinitionSource,
    ) -> Result<(), GeometryContractError> {
        let reference = match self {
            Self::FeatureResult(reference) | Self::BoundaryIntroducedBy(reference) => reference,
            _ => return Ok(()),
        };
        let graph = match source {
            GeometryDefinitionSource::Constructed(graph) => graph,
            GeometryDefinitionSource::Derived(derived) => &derived.graph,
            _ => {
                return Err(GeometryContractError::invalid(
                    "region feature selector",
                    "requires constructed or derived geometry",
                ))
            }
        };
        if graph
            .features
            .iter()
            .any(|feature| feature.feature_id == reference.0)
        {
            Ok(())
        } else {
            Err(GeometryContractError::invalid(
                "region feature selector",
                "must reference a declared feature",
            ))
        }
    }
}

impl GeometryDefinitionRef {
    fn validate(&self) -> Result<(), GeometryContractError> {
        token("geometry reference module path", &self.module_path)?;
        token("geometry reference id", &self.geometry_id)
    }
}

fn graph_source_must_differ(
    definition: &GeometryDefinition,
    source: &GeometryDefinitionRef,
) -> Result<(), GeometryContractError> {
    if definition.module_path == source.module_path && definition.geometry_id == source.geometry_id
    {
        Err(GeometryContractError::invalid(
            "derived geometry",
            "cannot derive from itself",
        ))
    } else {
        Ok(())
    }
}

pub(super) fn token(field: &str, value: &str) -> Result<(), GeometryContractError> {
    if value.trim().is_empty() {
        Err(GeometryContractError::invalid(field, "must not be empty"))
    } else {
        Ok(())
    }
}

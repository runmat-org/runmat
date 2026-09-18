use std::collections::BTreeMap;
use std::fmt::Display;

use runmat_geometry_core::{
    GeneratedGeometryDefinition, GeometryAuthoringValue, GeometryDefinition, GeometryDefinitionRef,
    GeometryDefinitionSource, GeometryDigest, ImportedGeometryDefinition,
};

use super::{
    execute_feature_plan, plan_feature_graph, resolve_geometry_parameters,
    ExecutedGeometryFeaturePlan, GeometryFeatureError, GeometryFeaturePlanLimits,
    GeometryFeatureProvider, GeometryFeatureResourceBounds, GeometryTopologyHistory,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedGeneratedGeometryRequest {
    pub module_path: String,
    pub function: String,
    pub source_digest: GeometryDigest,
    pub arguments: BTreeMap<String, GeometryAuthoringValue>,
}

pub trait GeometryMaterializationHost<G> {
    type Error: Display;

    fn import_exact(&self, source: &ImportedGeometryDefinition) -> Result<G, Self::Error>;

    fn generate(&self, request: &ResolvedGeneratedGeometryRequest) -> Result<G, Self::Error>;
}

pub struct GeometryDependency<'a, G> {
    pub reference: &'a GeometryDefinitionRef,
    pub geometry: &'a G,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryMaterializationKind {
    Imported,
    Constructed,
    Derived,
    Generated,
}

pub struct MaterializedGeometry<G> {
    pub definition: GeometryDefinitionRef,
    pub kind: GeometryMaterializationKind,
    pub geometry: G,
    pub feature_history: BTreeMap<String, GeometryTopologyHistory>,
    pub feature_resources: Option<GeometryFeatureResourceBounds>,
}

pub fn materialize_geometry_definition<P, H>(
    definition: &GeometryDefinition,
    bindings: &BTreeMap<String, GeometryAuthoringValue>,
    dependency: Option<GeometryDependency<'_, P::Geometry>>,
    provider: &P,
    host: &H,
    limits: GeometryFeaturePlanLimits,
) -> Result<MaterializedGeometry<P::Geometry>, GeometryFeatureError>
where
    P: GeometryFeatureProvider,
    H: GeometryMaterializationHost<P::Geometry>,
{
    definition
        .validate()
        .map_err(|error| GeometryFeatureError::InvalidPlan(error.to_string()))?;
    let parameters = resolve_geometry_parameters(&definition.parameters, bindings)?;
    let identity = GeometryDefinitionRef {
        module_path: definition.module_path.clone(),
        geometry_id: definition.geometry_id.clone(),
    };

    match &definition.source {
        GeometryDefinitionSource::Imported(source) => {
            reject_dependency(dependency.as_ref(), "imported")?;
            let geometry =
                host.import_exact(source)
                    .map_err(|error| GeometryFeatureError::Provider {
                        feature_id: definition.geometry_id.clone(),
                        message: error.to_string(),
                    })?;
            Ok(MaterializedGeometry {
                definition: identity,
                kind: GeometryMaterializationKind::Imported,
                geometry,
                feature_history: BTreeMap::new(),
                feature_resources: None,
            })
        }
        GeometryDefinitionSource::Constructed(graph) => {
            reject_dependency(dependency.as_ref(), "constructed")?;
            materialize_graph(
                identity,
                GeometryMaterializationKind::Constructed,
                graph,
                &parameters,
                None,
                provider,
                limits,
            )
        }
        GeometryDefinitionSource::Derived(derived) => {
            let dependency = dependency.ok_or_else(|| {
                GeometryFeatureError::InvalidPlan(format!(
                    "derived geometry `{}` requires source `{}::{}`",
                    definition.geometry_id, derived.source.module_path, derived.source.geometry_id
                ))
            })?;
            if dependency.reference != &derived.source {
                return Err(GeometryFeatureError::InvalidPlan(format!(
                    "derived geometry source does not match `{}::{}`",
                    derived.source.module_path, derived.source.geometry_id
                )));
            }
            materialize_graph(
                identity,
                GeometryMaterializationKind::Derived,
                &derived.graph,
                &parameters,
                Some(dependency.geometry),
                provider,
                limits,
            )
        }
        GeometryDefinitionSource::Generated(generated) => {
            reject_dependency(dependency.as_ref(), "generated")?;
            let request = generated_request(generated, &parameters)?;
            let geometry =
                host.generate(&request)
                    .map_err(|error| GeometryFeatureError::Provider {
                        feature_id: definition.geometry_id.clone(),
                        message: error.to_string(),
                    })?;
            Ok(MaterializedGeometry {
                definition: identity,
                kind: GeometryMaterializationKind::Generated,
                geometry,
                feature_history: BTreeMap::new(),
                feature_resources: None,
            })
        }
    }
}

fn materialize_graph<P: GeometryFeatureProvider>(
    definition: GeometryDefinitionRef,
    kind: GeometryMaterializationKind,
    graph: &runmat_geometry_core::GeometryFeatureGraph,
    parameters: &super::ResolvedGeometryParameters,
    source: Option<&P::Geometry>,
    provider: &P,
    limits: GeometryFeaturePlanLimits,
) -> Result<MaterializedGeometry<P::Geometry>, GeometryFeatureError> {
    let plan = plan_feature_graph(graph, parameters, &provider.capabilities(), limits)?;
    let resources = plan.resource_bounds;
    let ExecutedGeometryFeaturePlan {
        geometry,
        history_by_feature,
    } = execute_feature_plan(provider, &plan, source)?;
    Ok(MaterializedGeometry {
        definition,
        kind,
        geometry,
        feature_history: history_by_feature,
        feature_resources: Some(resources),
    })
}

fn generated_request(
    generated: &GeneratedGeometryDefinition,
    parameters: &super::ResolvedGeometryParameters,
) -> Result<ResolvedGeneratedGeometryRequest, GeometryFeatureError> {
    let arguments = generated
        .arguments
        .iter()
        .map(|(name, value)| Ok((name.clone(), parameters.resolve_value(value)?)))
        .collect::<Result<_, GeometryFeatureError>>()?;
    Ok(ResolvedGeneratedGeometryRequest {
        module_path: generated.module_path.clone(),
        function: generated.function.clone(),
        source_digest: generated.source_digest,
        arguments,
    })
}

fn reject_dependency<G>(
    dependency: Option<&GeometryDependency<'_, G>>,
    kind: &str,
) -> Result<(), GeometryFeatureError> {
    if dependency.is_some() {
        Err(GeometryFeatureError::InvalidPlan(format!(
            "{kind} geometry cannot receive a derived-source artifact"
        )))
    } else {
        Ok(())
    }
}

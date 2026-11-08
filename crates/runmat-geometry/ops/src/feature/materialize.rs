use std::collections::BTreeMap;
use std::fmt::Display;

use runmat_geometry_core::{
    GeneratedGeometryArgument, GeneratedGeometryDefinition, GeneratedGeometryProduct,
    GeneratedGeometryRegionOutput, GeometryAuthoringValue, GeometryDefinition,
    GeometryDefinitionRef, GeometryDefinitionSource, GeometryDigest, ImportedGeometryDefinition,
};

use super::{
    execute_feature_plan, plan_feature_graph, resolve_geometry_parameters,
    ExecutedGeometryFeaturePlan, GeometryFeatureError, GeometryFeaturePlanLimits,
    GeometryFeatureProvider, GeometryFeatureResourceBounds, GeometryTopologyHistory,
};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedGeneratedGeometryRequest {
    pub module_path: String,
    pub function: String,
    pub source_digest: GeometryDigest,
    pub arguments: Vec<GeneratedGeometryArgument>,
}

pub trait GeometryMaterializationHost<G> {
    type Error: Display;

    fn import_exact(&self, source: &ImportedGeometryDefinition) -> Result<G, Self::Error>;

    fn generate(
        &self,
        request: &ResolvedGeneratedGeometryRequest,
    ) -> Result<GeneratedGeometryProduct, Self::Error>;
}

pub struct GeometryDependency<'a, G> {
    pub reference: &'a GeometryDefinitionRef,
    pub geometry: &'a G,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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
    pub generated_region_outputs: BTreeMap<String, GeneratedGeometryRegionOutput>,
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
                generated_region_outputs: BTreeMap::new(),
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
        GeometryDefinitionSource::CrossSection(_) => Err(GeometryFeatureError::InvalidPlan(
            "a cross-section is meshed and solved by the transmission-line solver; no geometry kernel materializes it".into(),
        )),
        GeometryDefinitionSource::Generated(generated) => {
            reject_dependency(dependency.as_ref(), "generated")?;
            let request =
                resolve_generated_geometry_request(generated, &definition.parameters, bindings)?;
            let product =
                host.generate(&request)
                    .map_err(|error| GeometryFeatureError::Provider {
                        feature_id: definition.geometry_id.clone(),
                        message: error.to_string(),
                    })?;
            product
                .validate()
                .map_err(|error| GeometryFeatureError::InvalidPlan(error.to_string()))?;
            let outputs = product.region_outputs;
            let generated_parameters = resolve_geometry_parameters(&[], &BTreeMap::new())?;
            let mut materialized = materialize_graph(
                identity,
                GeometryMaterializationKind::Generated,
                &product.graph,
                &generated_parameters,
                None,
                provider,
                limits,
            )?;
            materialized.generated_region_outputs = outputs;
            Ok(materialized)
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
        generated_region_outputs: BTreeMap::new(),
    })
}

pub fn resolve_generated_geometry_request(
    generated: &GeneratedGeometryDefinition,
    parameter_definitions: &[runmat_geometry_core::GeometryParameterDefinition],
    bindings: &BTreeMap<String, GeometryAuthoringValue>,
) -> Result<ResolvedGeneratedGeometryRequest, GeometryFeatureError> {
    let parameters = resolve_geometry_parameters(parameter_definitions, bindings)?;
    let arguments = generated
        .arguments
        .iter()
        .map(|argument| {
            Ok(GeneratedGeometryArgument {
                parameter: argument.parameter.clone(),
                value: parameters.resolve_value(&argument.value)?,
            })
        })
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

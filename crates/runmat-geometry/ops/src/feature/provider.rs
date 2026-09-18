use std::collections::BTreeMap;
use std::fmt::Display;

use super::{
    ExecutedGeometryFeaturePlan, GeometryFeatureCapabilities, GeometryFeatureError,
    GeometryFeaturePlan, GeometryFeatureProviderOutput, ResolvedGeometryFeatureOperation,
};

pub trait GeometryFeatureProvider {
    type Geometry;
    type Error: Display;

    fn capabilities(&self) -> GeometryFeatureCapabilities;

    fn execute(
        &self,
        feature_id: &str,
        operation: &ResolvedGeometryFeatureOperation,
        dependencies: &[(&str, &Self::Geometry)],
        source: Option<&Self::Geometry>,
    ) -> Result<GeometryFeatureProviderOutput<Self::Geometry>, Self::Error>;
}

pub fn execute_feature_plan<P: GeometryFeatureProvider>(
    provider: &P,
    plan: &GeometryFeaturePlan,
    source: Option<&P::Geometry>,
) -> Result<ExecutedGeometryFeaturePlan<P::Geometry>, GeometryFeatureError> {
    let capabilities = provider.capabilities();
    let mut values = BTreeMap::<String, P::Geometry>::new();
    let mut history_by_feature = BTreeMap::new();

    for step in &plan.steps {
        if !capabilities.supports(step.operation.kind()) {
            return Err(GeometryFeatureError::UnsupportedOperation {
                feature_id: step.feature_id.clone(),
                kind: step.operation.kind(),
            });
        }
        let dependencies = step
            .dependency_ids
            .iter()
            .map(|reference| {
                if reference.0 == "source.result" {
                    source
                        .map(|value| (reference.0.as_str(), value))
                        .ok_or_else(|| {
                            GeometryFeatureError::InvalidPlan(format!(
                                "feature `{}` requires source geometry",
                                step.feature_id
                            ))
                        })
                } else {
                    values
                        .get(&reference.0)
                        .map(|value| (reference.0.as_str(), value))
                        .ok_or_else(|| {
                            GeometryFeatureError::InvalidPlan(format!(
                                "feature `{}` references unavailable dependency `{}`",
                                step.feature_id, reference.0
                            ))
                        })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = provider
            .execute(&step.feature_id, &step.operation, &dependencies, source)
            .map_err(|error| GeometryFeatureError::Provider {
                feature_id: step.feature_id.clone(),
                message: error.to_string(),
            })?;
        output.history.validate()?;
        if values
            .insert(step.feature_id.clone(), output.geometry)
            .is_some()
        {
            return Err(GeometryFeatureError::InvalidPlan(format!(
                "duplicate feature `{}`",
                step.feature_id
            )));
        }
        history_by_feature.insert(step.feature_id.clone(), output.history);
    }

    let geometry = values.remove(&plan.result.0).ok_or_else(|| {
        GeometryFeatureError::InvalidPlan(format!(
            "result feature `{}` was not produced",
            plan.result.0
        ))
    })?;
    Ok(ExecutedGeometryFeaturePlan {
        geometry,
        history_by_feature,
    })
}

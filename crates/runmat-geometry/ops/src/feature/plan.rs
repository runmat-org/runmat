use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{GeometryFeatureGraph, GeometryFeatureOperation, GeometryFeatureRef};

use super::{
    GeometryFeatureCapabilities, GeometryFeatureError, GeometryFeaturePlan,
    GeometryFeaturePlanStep, GeometryFeatureResourceBounds, ResolvedGeometryFeatureOperation,
    ResolvedGeometryParameters,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryFeaturePlanLimits {
    pub maximum_features: u64,
    pub maximum_dependency_edges: u64,
}

impl Default for GeometryFeaturePlanLimits {
    fn default() -> Self {
        Self {
            maximum_features: 10_000,
            maximum_dependency_edges: 100_000,
        }
    }
}

pub fn plan_feature_graph(
    graph: &GeometryFeatureGraph,
    parameters: &ResolvedGeometryParameters,
    capabilities: &GeometryFeatureCapabilities,
    limits: GeometryFeaturePlanLimits,
) -> Result<GeometryFeaturePlan, GeometryFeatureError> {
    let feature_count = u64::try_from(graph.features.len())
        .map_err(|_| GeometryFeatureError::InvalidPlan("feature count overflow".into()))?;
    let maximum_features = limits.maximum_features.min(capabilities.maximum_features);
    if feature_count == 0 || feature_count > maximum_features {
        return Err(GeometryFeatureError::InvalidPlan(format!(
            "feature count {feature_count} is outside the admitted range 1..={maximum_features}"
        )));
    }

    let mut steps = Vec::with_capacity(graph.features.len());
    let mut dependency_edge_count = 0_u64;
    let mut available = BTreeSet::<String>::new();
    for feature in &graph.features {
        if available.contains(&feature.feature_id) {
            return Err(GeometryFeatureError::InvalidPlan(format!(
                "duplicate feature `{}`",
                feature.feature_id
            )));
        }
        let operation = resolve_operation(&feature.operation, parameters)?;
        if !capabilities.supports(operation.kind()) {
            return Err(GeometryFeatureError::UnsupportedOperation {
                feature_id: feature.feature_id.clone(),
                kind: operation.kind(),
            });
        }
        validate_provider_bounds(&feature.feature_id, &operation, capabilities)?;
        let dependency_ids = dependencies(&operation);
        for dependency in &dependency_ids {
            if dependency.0 == feature.feature_id {
                return Err(GeometryFeatureError::InvalidPlan(format!(
                    "feature `{}` cannot depend on itself",
                    feature.feature_id
                )));
            }
            if dependency.0 != "source.result" && !available.contains(&dependency.0) {
                return Err(GeometryFeatureError::InvalidPlan(format!(
                    "feature `{}` references unavailable dependency `{}`",
                    feature.feature_id, dependency.0
                )));
            }
        }
        dependency_edge_count = dependency_edge_count
            .checked_add(u64::try_from(dependency_ids.len()).map_err(|_| {
                GeometryFeatureError::InvalidPlan("dependency count overflow".into())
            })?)
            .ok_or_else(|| GeometryFeatureError::InvalidPlan("dependency count overflow".into()))?;
        if dependency_edge_count > limits.maximum_dependency_edges {
            return Err(GeometryFeatureError::InvalidPlan(format!(
                "dependency edge count exceeds {}",
                limits.maximum_dependency_edges
            )));
        }
        steps.push(GeometryFeaturePlanStep {
            feature_id: feature.feature_id.clone(),
            operation,
            dependency_ids,
        });
        available.insert(feature.feature_id.clone());
    }

    if !available.contains(&graph.result.0) {
        return Err(GeometryFeatureError::InvalidPlan(format!(
            "result feature `{}` is unavailable",
            graph.result.0
        )));
    }

    let maximum_live_artifacts = maximum_live_artifacts(&steps, &graph.result)?;
    Ok(GeometryFeaturePlan {
        steps,
        result: graph.result.clone(),
        resource_bounds: GeometryFeatureResourceBounds {
            feature_count,
            dependency_edge_count,
            provider_operation_count: feature_count,
            maximum_live_artifacts,
        },
    })
}

fn resolve_operation(
    operation: &GeometryFeatureOperation,
    parameters: &ResolvedGeometryParameters,
) -> Result<ResolvedGeometryFeatureOperation, GeometryFeatureError> {
    use GeometryFeatureOperation as Source;
    use ResolvedGeometryFeatureOperation as Target;
    let resolved = match operation {
        Source::Box {
            width_m,
            depth_m,
            height_m,
        } => Target::Box {
            width_m: runmat_types::physical::Length::new(positive(
                parameters.length(width_m)?,
                "box width",
            )?),
            depth_m: runmat_types::physical::Length::new(positive(
                parameters.length(depth_m)?,
                "box depth",
            )?),
            height_m: runmat_types::physical::Length::new(positive(
                parameters.length(height_m)?,
                "box height",
            )?),
        },
        Source::Cylinder {
            radius_m,
            height_m,
            axis,
            origin_m,
        } => Target::Cylinder {
            radius_m: runmat_types::physical::Length::new(positive(
                parameters.length(radius_m)?,
                "cylinder radius",
            )?),
            height_m: runmat_types::physical::Length::new(positive(
                parameters.length(height_m)?,
                "cylinder height",
            )?),
            axis: direction(parameters.vector(axis)?, "cylinder axis")?,
            origin_m: parameters.position(origin_m)?,
        },
        Source::Sphere { radius_m, center_m } => Target::Sphere {
            radius_m: runmat_types::physical::Length::new(positive(
                parameters.length(radius_m)?,
                "sphere radius",
            )?),
            center_m: parameters.position(center_m)?,
        },
        Source::Transform { source, matrix } => Target::Transform {
            source: source.clone(),
            matrix: parameters.matrix(matrix)?,
        },
        Source::Instance { source, transforms } => Target::Instance {
            source: source.clone(),
            transforms: transforms
                .iter()
                .map(|matrix| parameters.matrix(matrix))
                .collect::<Result<_, _>>()?,
        },
        Source::Extrude {
            profile,
            distance_m,
            direction: axis,
        } => Target::Extrude {
            profile: profile.clone(),
            distance_m: runmat_types::physical::Length::new(positive(
                parameters.length(distance_m)?,
                "extrusion distance",
            )?),
            direction: direction(parameters.vector(axis)?, "extrusion direction")?,
        },
        Source::Revolve {
            profile,
            axis,
            origin_m,
            angle_rad,
        } => Target::Revolve {
            profile: profile.clone(),
            axis: direction(parameters.vector(axis)?, "revolution axis")?,
            origin_m: parameters.position(origin_m)?,
            angle_rad: runmat_types::physical::Angle::new(positive(
                parameters.angle(angle_rad)?,
                "revolution angle",
            )?),
        },
        Source::Union { operands } => Target::Union {
            operands: operands.clone(),
        },
        Source::Subtract { base, tools } => Target::Subtract {
            base: base.clone(),
            tools: tools.clone(),
        },
        Source::Intersect { operands } => Target::Intersect {
            operands: operands.clone(),
        },
        Source::Hole {
            target,
            radius_m,
            depth_m,
            axis,
            origin_m,
        } => Target::Hole {
            target: target.clone(),
            radius_m: runmat_types::physical::Length::new(positive(
                parameters.length(radius_m)?,
                "hole radius",
            )?),
            depth_m: runmat_types::physical::Length::new(positive(
                parameters.length(depth_m)?,
                "hole depth",
            )?),
            axis: direction(parameters.vector(axis)?, "hole axis")?,
            origin_m: parameters.position(origin_m)?,
        },
        Source::Pocket {
            target,
            profile,
            depth_m,
            direction: axis,
        } => Target::Pocket {
            target: target.clone(),
            profile: profile.clone(),
            depth_m: runmat_types::physical::Length::new(positive(
                parameters.length(depth_m)?,
                "pocket depth",
            )?),
            direction: direction(parameters.vector(axis)?, "pocket direction")?,
        },
        Source::Fillet {
            target,
            edges,
            radius_m,
        } => Target::Fillet {
            target: target.clone(),
            edges: edges.clone(),
            radius_m: runmat_types::physical::Length::new(positive(
                parameters.length(radius_m)?,
                "fillet radius",
            )?),
        },
        Source::Chamfer {
            target,
            edges,
            distance_m,
        } => Target::Chamfer {
            target: target.clone(),
            edges: edges.clone(),
            distance_m: runmat_types::physical::Length::new(positive(
                parameters.length(distance_m)?,
                "chamfer distance",
            )?),
        },
        Source::RegenerateSource { parameters: values } => Target::RegenerateSource {
            parameters: values
                .iter()
                .map(|(name, value)| Ok((name.clone(), parameters.resolve_value(value)?)))
                .collect::<Result<_, GeometryFeatureError>>()?,
        },
    };
    Ok(resolved)
}

fn dependencies(operation: &ResolvedGeometryFeatureOperation) -> Vec<GeometryFeatureRef> {
    use ResolvedGeometryFeatureOperation as Operation;
    match operation {
        Operation::Transform { source, .. } | Operation::Instance { source, .. } => {
            vec![source.clone()]
        }
        Operation::Extrude { profile, .. } | Operation::Revolve { profile, .. } => {
            vec![profile.clone()]
        }
        Operation::Union { operands } | Operation::Intersect { operands } => operands.clone(),
        Operation::Subtract { base, tools } => std::iter::once(base.clone())
            .chain(tools.iter().cloned())
            .collect(),
        Operation::Hole { target, .. } => vec![target.clone()],
        Operation::Pocket {
            target, profile, ..
        } => vec![target.clone(), profile.clone()],
        Operation::Fillet { target, .. } | Operation::Chamfer { target, .. } => {
            vec![target.clone()]
        }
        Operation::Box { .. }
        | Operation::Cylinder { .. }
        | Operation::Sphere { .. }
        | Operation::RegenerateSource { .. } => Vec::new(),
    }
}

fn validate_provider_bounds(
    feature_id: &str,
    operation: &ResolvedGeometryFeatureOperation,
    capabilities: &GeometryFeatureCapabilities,
) -> Result<(), GeometryFeatureError> {
    let count = match operation {
        ResolvedGeometryFeatureOperation::Instance { transforms, .. } => {
            let count = u64::try_from(transforms.len()).unwrap_or(u64::MAX);
            if count > capabilities.maximum_instances_per_feature {
                return Err(GeometryFeatureError::InvalidPlan(format!(
                    "feature `{feature_id}` exceeds the provider instance bound"
                )));
            }
            return Ok(());
        }
        ResolvedGeometryFeatureOperation::Union { operands }
        | ResolvedGeometryFeatureOperation::Intersect { operands } => operands.len(),
        ResolvedGeometryFeatureOperation::Subtract { tools, .. } => tools.len() + 1,
        _ => return Ok(()),
    };
    if u64::try_from(count).unwrap_or(u64::MAX) > capabilities.maximum_boolean_operands {
        Err(GeometryFeatureError::InvalidPlan(format!(
            "feature `{feature_id}` exceeds the provider boolean-operand bound"
        )))
    } else {
        Ok(())
    }
}

fn maximum_live_artifacts(
    steps: &[GeometryFeaturePlanStep],
    result: &GeometryFeatureRef,
) -> Result<u64, GeometryFeatureError> {
    let mut remaining_uses = BTreeMap::<&str, u64>::new();
    for step in steps {
        for dependency in &step.dependency_ids {
            if dependency.0 != "source.result" {
                *remaining_uses.entry(&dependency.0).or_default() += 1;
            }
        }
    }
    *remaining_uses.entry(&result.0).or_default() += 1;
    let mut live = 0_u64;
    let mut maximum = 0_u64;
    for step in steps {
        live = live.checked_add(1).ok_or_else(|| {
            GeometryFeatureError::InvalidPlan("live artifact count overflow".into())
        })?;
        maximum = maximum.max(live);
        for dependency in &step.dependency_ids {
            if dependency.0 == "source.result" {
                continue;
            }
            let uses = remaining_uses
                .get_mut(dependency.0.as_str())
                .ok_or_else(|| {
                    GeometryFeatureError::InvalidPlan(format!(
                        "unknown dependency `{}`",
                        dependency.0
                    ))
                })?;
            *uses = uses.checked_sub(1).ok_or_else(|| {
                GeometryFeatureError::InvalidPlan("dependency use underflow".into())
            })?;
            if *uses == 0 {
                live = live.checked_sub(1).ok_or_else(|| {
                    GeometryFeatureError::InvalidPlan("live artifact count underflow".into())
                })?;
            }
        }
        if remaining_uses
            .get(step.feature_id.as_str())
            .copied()
            .unwrap_or(0)
            == 0
        {
            live = live.checked_sub(1).ok_or_else(|| {
                GeometryFeatureError::InvalidPlan("live artifact count underflow".into())
            })?;
        }
    }
    Ok(maximum)
}

fn positive(value: f64, field: &str) -> Result<f64, GeometryFeatureError> {
    if value > 0.0 {
        Ok(value)
    } else {
        Err(GeometryFeatureError::InvalidPlan(format!(
            "{field} must be positive"
        )))
    }
}

fn direction(value: [f64; 3], field: &str) -> Result<[f64; 3], GeometryFeatureError> {
    let norm_squared = value
        .iter()
        .map(|component| component * component)
        .sum::<f64>();
    if norm_squared > 0.0 && norm_squared.is_finite() {
        Ok(value)
    } else {
        Err(GeometryFeatureError::InvalidPlan(format!(
            "{field} must be finite and nonzero"
        )))
    }
}

use super::value;
use crate::{
    CheckedCallArgument, CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration,
    CheckedFeaStatement, CheckedFeaStatementKind, FeaDiagnostic, FeaType, PhysicalDimension,
};
use runmat_geometry_core::{
    GeometryFeature, GeometryFeatureGraph, GeometryFeatureOperation, GeometryFeatureRef,
    GeometryFeatureRegionSelection, GeometryRegionDefinition, GeometryRegionRef,
    GeometryRegionRole,
};
use std::collections::BTreeMap;

pub(super) fn graph(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    regions: &[GeometryRegionDefinition],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryFeatureGraph> {
    let initial_errors = diagnostics.len();
    let mut features = Vec::new();
    let mut result = None;
    let mut source_parameters = BTreeMap::new();
    for statement in statements {
        match &statement.kind {
            CheckedFeaStatementKind::Feature { name, value } => {
                let operation =
                    operation(declaration, value, regions, &BTreeMap::new(), diagnostics)?;
                features.push(GeometryFeature {
                    feature_id: name.clone(),
                    operation,
                });
            }
            CheckedFeaStatementKind::SourceParameterUpdate { target, value } => {
                let [source, parameter, name] = target.as_slice() else {
                    super::error(
                        declaration,
                        statement.span,
                        "source parameter update requires `set source.parameter.<name> = <value>`",
                        diagnostics,
                    );
                    continue;
                };
                if source != "source" || parameter != "parameter" {
                    super::error(
                        declaration,
                        statement.span,
                        "source parameter update requires `set source.parameter.<name> = <value>`",
                        diagnostics,
                    );
                    continue;
                }
                let Some(value) = value::authoring_value(value) else {
                    super::error(
                        declaration,
                        statement.span,
                        "source parameter update requires a canonical typed value",
                        diagnostics,
                    );
                    continue;
                };
                if source_parameters.insert(name.clone(), value).is_some() {
                    super::error(
                        declaration,
                        statement.span,
                        format!("source parameter `{name}` is updated more than once"),
                        diagnostics,
                    );
                }
            }
            CheckedFeaStatementKind::Assignment { target, value, .. }
                if target.as_slice() == ["result"] =>
            {
                if result.is_some() {
                    super::error(
                        declaration,
                        statement.span,
                        "geometry feature block declares `result` more than once",
                        diagnostics,
                    );
                    continue;
                }
                result = result_expression(
                    declaration,
                    value,
                    regions,
                    &source_parameters,
                    &mut features,
                    diagnostics,
                );
            }
            CheckedFeaStatementKind::Input(_) => {}
            _ => super::error(
                declaration,
                statement.span,
                "geometry feature blocks accept only `feature` declarations and one `result`",
                diagnostics,
            ),
        }
    }
    if result.is_none() {
        super::error(
            declaration,
            declaration.span,
            "geometry feature block requires exactly one `result`",
            diagnostics,
        );
    }
    if !source_parameters.is_empty()
        && !features.iter().any(|feature| {
            matches!(
                &feature.operation,
                GeometryFeatureOperation::RegenerateSource { .. }
            )
        })
    {
        super::error(
            declaration,
            declaration.span,
            "source parameter updates require `result = regenerate(source)`",
            diagnostics,
        );
    }
    (diagnostics.len() == initial_errors).then(|| GeometryFeatureGraph {
        features,
        result: result.expect("checked above"),
    })
}

fn result_expression(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    regions: &[GeometryRegionDefinition],
    source_parameters: &BTreeMap<String, runmat_geometry_core::GeometryAuthoringValue>,
    features: &mut Vec<GeometryFeature>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryFeatureRef> {
    if let Some(reference) = feature_ref(value) {
        return Some(reference);
    }
    let operation = operation(declaration, value, regions, source_parameters, diagnostics)?;
    if features
        .iter()
        .any(|feature| feature.feature_id == "result")
    {
        super::error(
            declaration,
            value.span,
            "inline geometry result conflicts with feature named `result`",
            diagnostics,
        );
        return None;
    }
    features.push(GeometryFeature {
        feature_id: "result".to_string(),
        operation,
    });
    Some(GeometryFeatureRef("result".to_string()))
}

fn operation(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    regions: &[GeometryRegionDefinition],
    source_parameters: &BTreeMap<String, runmat_geometry_core::GeometryAuthoringValue>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryFeatureOperation> {
    let result = operation_inner(declaration, value, regions, source_parameters);
    if result.is_none() {
        super::error(
            declaration,
            value.span,
            "feature value must be a valid typed geometry operation",
            diagnostics,
        );
    }
    result
}

fn operation_inner(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    regions: &[GeometryRegionDefinition],
    source_parameters: &BTreeMap<String, runmat_geometry_core::GeometryAuthoringValue>,
) -> Option<GeometryFeatureOperation> {
    let CheckedExpressionKind::OperationCall {
        lowering,
        arguments,
        ..
    } = &value.kind
    else {
        return None;
    };
    let name = lowering
        .as_str()
        .strip_prefix("fea.geometry.")
        .unwrap_or_default();
    Some(match name {
        "box" => GeometryFeatureOperation::Box {
            width_m: length(argument(arguments, 0, "width")?)?,
            depth_m: length(argument(arguments, 1, "depth")?)?,
            height_m: length(argument(arguments, 2, "height")?)?,
        },
        "cylinder" => GeometryFeatureOperation::Cylinder {
            radius_m: length(argument(arguments, 0, "radius")?)?,
            height_m: length(argument(arguments, 1, "height")?)?,
            axis: value::vector3(argument(arguments, 2, "axis")?)?,
            origin_m: length_vector(argument(arguments, 3, "origin")?)?,
        },
        "sphere" => GeometryFeatureOperation::Sphere {
            radius_m: length(argument(arguments, 0, "radius")?)?,
            center_m: length_vector(argument(arguments, 1, "center")?)?,
        },
        "transform" => GeometryFeatureOperation::Transform {
            source: feature_ref(argument(arguments, 0, "source")?)?,
            matrix: matrix(argument(arguments, 1, "matrix")?)?,
        },
        "instance" => GeometryFeatureOperation::Instance {
            source: feature_ref(argument(arguments, 0, "source")?)?,
            transforms: matrices(argument(arguments, 1, "transforms")?)?,
        },
        "extrude" => GeometryFeatureOperation::Extrude {
            profile: feature_ref(argument(arguments, 0, "profile")?)?,
            distance_m: length(argument(arguments, 1, "distance")?)?,
            direction: value::vector3(argument(arguments, 2, "direction")?)?,
        },
        "revolve" => GeometryFeatureOperation::Revolve {
            profile: feature_ref(argument(arguments, 0, "profile")?)?,
            axis: value::vector3(argument(arguments, 1, "axis")?)?,
            origin_m: length_vector(argument(arguments, 2, "origin")?)?,
            angle_rad: angle(argument(arguments, 3, "angle")?)?,
        },
        "union" => GeometryFeatureOperation::Union {
            operands: feature_refs(arguments)?,
        },
        "subtract" => {
            let mut operands = feature_refs(arguments)?.into_iter();
            GeometryFeatureOperation::Subtract {
                base: operands.next()?,
                tools: operands.collect(),
            }
        }
        "intersect" => GeometryFeatureOperation::Intersect {
            operands: feature_refs(arguments)?,
        },
        "hole" => GeometryFeatureOperation::Hole {
            target: feature_ref(argument(arguments, 0, "target")?)?,
            radius_m: length(argument(arguments, 1, "radius")?)?,
            depth_m: length(argument(arguments, 2, "depth")?)?,
            axis: value::vector3(argument(arguments, 3, "axis")?)?,
            origin_m: length_vector(argument(arguments, 4, "origin")?)?,
        },
        "pocket" => GeometryFeatureOperation::Pocket {
            target: feature_ref(argument(arguments, 0, "target")?)?,
            profile: feature_ref(argument(arguments, 1, "profile")?)?,
            depth_m: length(argument(arguments, 2, "depth")?)?,
            direction: value::vector3(argument(arguments, 3, "direction")?)?,
        },
        "fillet" => GeometryFeatureOperation::Fillet {
            target: feature_ref(argument(arguments, 0, "target")?)?,
            edges: region_selection(declaration, regions, argument(arguments, 1, "edges")?)?,
            radius_m: length(argument(arguments, 2, "radius")?)?,
        },
        "chamfer" => GeometryFeatureOperation::Chamfer {
            target: feature_ref(argument(arguments, 0, "target")?)?,
            edges: region_selection(declaration, regions, argument(arguments, 1, "edges")?)?,
            distance_m: length(argument(arguments, 2, "distance")?)?,
        },
        "regenerate" => {
            let source = feature_ref(argument(arguments, 0, "source")?)?;
            if source.0 != "source.result" || source_parameters.is_empty() {
                return None;
            }
            GeometryFeatureOperation::RegenerateSource {
                parameters: source_parameters.clone(),
            }
        }
        _ => return None,
    })
}

fn argument<'a>(
    arguments: &'a [CheckedCallArgument],
    position: usize,
    name: &str,
) -> Option<&'a CheckedExpression> {
    arguments
        .iter()
        .find_map(|argument| match argument {
            CheckedCallArgument::Named {
                name: candidate,
                value,
                ..
            } if candidate == name => Some(value),
            _ => None,
        })
        .or_else(|| arguments.get(position).map(argument_value))
}

fn argument_value(argument: &CheckedCallArgument) -> &CheckedExpression {
    match argument {
        CheckedCallArgument::Positional(value) | CheckedCallArgument::Named { value, .. } => value,
    }
}

fn length(value: &CheckedExpression) -> Option<runmat_geometry_core::GeometryScalarExpression> {
    ((value.value_type).is_quantity((PhysicalDimension::LENGTH).into()))
        .then(|| value::scalar(value))?
}

fn angle(value: &CheckedExpression) -> Option<runmat_geometry_core::GeometryScalarExpression> {
    (value
        .value_type
        .is_quantity(runmat_types::physical::DynamicDimension::quantity_type(
            runmat_types::physical::QuantityKind::Angle,
        )))
    .then(|| value::scalar(value))?
}

fn length_vector(
    value: &CheckedExpression,
) -> Option<[runmat_geometry_core::GeometryScalarExpression; 3]> {
    matches!(
        &value.value_type,
        FeaType::Vector {
            element,
            length: Some(3),
        } if element.is_quantity(PhysicalDimension::LENGTH.into())
    )
    .then(|| value::vector3(value))?
}

fn feature_ref(value: &CheckedExpression) -> Option<GeometryFeatureRef> {
    let CheckedExpressionKind::Reference(path) = &value.kind else {
        return None;
    };
    match path.as_slice() {
        [source] if source == "source" => Some(GeometryFeatureRef("source.result".to_string())),
        [namespace, name] if namespace == "feature" => Some(GeometryFeatureRef(name.clone())),
        [source, result] if source == "source" && result == "result" => {
            Some(GeometryFeatureRef("source.result".to_string()))
        }
        _ => None,
    }
}

fn feature_refs(arguments: &[CheckedCallArgument]) -> Option<Vec<GeometryFeatureRef>> {
    arguments
        .iter()
        .map(|argument| feature_ref(argument_value(argument)))
        .collect()
}

fn matrix(
    value: &CheckedExpression,
) -> Option<Vec<runmat_geometry_core::GeometryScalarExpression>> {
    let CheckedExpressionKind::List(values) = &value.kind else {
        return None;
    };
    (values.len() == 16).then(|| values.iter().map(value::scalar).collect::<Option<Vec<_>>>())?
}

fn matrices(
    value: &CheckedExpression,
) -> Option<Vec<Vec<runmat_geometry_core::GeometryScalarExpression>>> {
    let CheckedExpressionKind::List(values) = &value.kind else {
        return None;
    };
    values.iter().map(matrix).collect()
}

fn region_ref(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
) -> Option<GeometryRegionRef> {
    let CheckedExpressionKind::Reference(path) = &value.kind else {
        return None;
    };
    let [geometry, geometry_id, region, region_id] = path.as_slice() else {
        return None;
    };
    let FeaType::Region(role) = &value.value_type else {
        return None;
    };
    (geometry == "geometry" && region == "region").then(|| GeometryRegionRef {
        module_path: declaration.module.as_str().to_string(),
        geometry_id: geometry_id.clone(),
        region_id: region_id.clone(),
        role: match *role {
            crate::FeaTopologyRole::Vertex => GeometryRegionRole::Vertex,
            crate::FeaTopologyRole::Edge => GeometryRegionRole::Edge,
            crate::FeaTopologyRole::Face => GeometryRegionRole::Face,
            crate::FeaTopologyRole::Solid => GeometryRegionRole::Solid,
            crate::FeaTopologyRole::Cell => GeometryRegionRole::Cell,
        },
    })
}

fn region_selection(
    declaration: &CheckedFeaDeclaration,
    regions: &[GeometryRegionDefinition],
    value: &CheckedExpression,
) -> Option<GeometryFeatureRegionSelection> {
    let reference = region_ref(declaration, value)?;
    if reference.role != GeometryRegionRole::Edge
        || reference.module_path != declaration.module.as_str()
        || reference.geometry_id != declaration.key.name
    {
        return None;
    }
    let region = regions
        .iter()
        .find(|region| region.region_id == reference.region_id)?;
    Some(GeometryFeatureRegionSelection {
        role: region.role,
        selector: region.selector.clone(),
        cardinality: region.cardinality,
    })
}

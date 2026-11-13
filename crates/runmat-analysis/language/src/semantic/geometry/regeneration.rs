use std::collections::BTreeMap;

use runmat_geometry_core::{
    Ap242ParameterAccess, CadParameterAccess, CadParameterPhysicalType, GeometryDefinition,
    GeometryDefinitionSource, GeometryFeatureOperation,
};

use crate::{
    CheckedFeaDeclaration, CheckedFeaStatementKind, FeaDiagnostic, FeaSymbolKey, FeaSymbolKind,
    FeaType, LogicalSourcePath, PhysicalDimension,
};

pub(super) fn remove_unqualified(
    geometries: &mut BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition>,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let invalid = geometries
        .iter()
        .filter_map(|(identity, definition)| {
            let GeometryDefinitionSource::Derived(derived) = &definition.source else {
                return None;
            };
            let parameters = derived.graph.features.iter().find_map(|feature| {
                if let GeometryFeatureOperation::RegenerateSource { parameters } =
                    &feature.operation
                {
                    Some(parameters)
                } else {
                    None
                }
            })?;
            let declaration = declarations.get(identity)?;
            qualification_error(geometries, derived, declaration, parameters)
                .map(|message| (identity.clone(), message))
        })
        .collect::<Vec<_>>();
    for (identity, message) in invalid {
        if let Some(declaration) = declarations.get(&identity) {
            super::error(declaration, declaration.span, message, diagnostics);
        }
        geometries.remove(&identity);
    }
}

fn qualification_error(
    geometries: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition>,
    derived: &runmat_geometry_core::DerivedGeometryDefinition,
    declaration: &CheckedFeaDeclaration,
    parameters: &BTreeMap<String, runmat_geometry_core::GeometryAuthoringValue>,
) -> Option<String> {
    let source_path = match LogicalSourcePath::new(&derived.source.module_path) {
        Ok(path) => path,
        Err(_) => return Some("source regeneration has an invalid source module path".to_string()),
    };
    let Some(source) = geometries.get(&(
        source_path,
        FeaSymbolKey {
            kind: FeaSymbolKind::Geometry,
            name: derived.source.geometry_id.clone(),
        },
    )) else {
        return Some("source regeneration cannot resolve its exact source geometry".to_string());
    };
    let GeometryDefinitionSource::Imported(imported) = &source.source else {
        return Some("source regeneration requires an imported exact-CAD source".to_string());
    };
    if !imported.capabilities.regeneration {
        return Some(
            "source regeneration requires a source with qualified regeneration capability"
                .to_string(),
        );
    }
    for parameter_id in parameters.keys() {
        let Some(parameter) = source
            .cad_parameters
            .iter()
            .find(|parameter| parameter.parameter_id == *parameter_id)
        else {
            return Some(format!(
                "source regeneration parameter `{parameter_id}` is not declared by the imported geometry"
            ));
        };
        if parameter.access != CadParameterAccess::Writable
            || parameter.source.access != Ap242ParameterAccess::Regenerable
        {
            return Some(format!(
                "source regeneration parameter `{parameter_id}` is not qualified writable/regenerable"
            ));
        }
        let Some(value_type) = update_type(declaration, parameter_id) else {
            return Some(format!(
                "source regeneration parameter `{parameter_id}` has no checked update value"
            ));
        };
        if !type_matches(parameter.physical_type, value_type) {
            return Some(format!(
                "source regeneration parameter `{parameter_id}` requires a {} value, not {}",
                type_name(parameter.physical_type),
                fea_type_name(value_type)
            ));
        }
    }
    None
}

fn update_type<'a>(
    declaration: &'a CheckedFeaDeclaration,
    parameter_id: &str,
) -> Option<&'a FeaType> {
    update_type_in(&declaration.statements, parameter_id)
}

fn update_type_in<'a>(
    statements: &'a [crate::CheckedFeaStatement],
    parameter_id: &str,
) -> Option<&'a FeaType> {
    statements
        .iter()
        .find_map(|statement| match &statement.kind {
            CheckedFeaStatementKind::SourceParameterUpdate { target, value }
                if target.as_slice() == ["source", "parameter", parameter_id] =>
            {
                Some(&value.value_type)
            }
            CheckedFeaStatementKind::Block { statements, .. } => {
                update_type_in(statements, parameter_id)
            }
            _ => None,
        })
}

fn type_matches(expected: CadParameterPhysicalType, actual: &FeaType) -> bool {
    match expected {
        CadParameterPhysicalType::Dimensionless => {
            (matches!(actual, FeaType::Integer | FeaType::Real)
                || actual.is_quantity(PhysicalDimension::DIMENSIONLESS.into()))
        }
        CadParameterPhysicalType::Angle => {
            actual.is_quantity(runmat_types::physical::DynamicDimension::quantity_type(
                runmat_types::physical::QuantityKind::Angle,
            ))
        }
        CadParameterPhysicalType::Length => {
            (actual).is_quantity((PhysicalDimension::LENGTH).into())
        }
        CadParameterPhysicalType::Area => (actual).is_quantity((PhysicalDimension::AREA).into()),
        CadParameterPhysicalType::Volume => {
            (actual).is_quantity((PhysicalDimension::VOLUME).into())
        }
        CadParameterPhysicalType::Mass => (actual).is_quantity((PhysicalDimension::MASS).into()),
        CadParameterPhysicalType::Temperature => {
            (actual).is_quantity((PhysicalDimension::TEMPERATURE).into())
        }
    }
}

fn type_name(value: CadParameterPhysicalType) -> &'static str {
    match value {
        CadParameterPhysicalType::Dimensionless => "dimensionless",
        CadParameterPhysicalType::Length => "length",
        CadParameterPhysicalType::Angle => "angle",
        CadParameterPhysicalType::Area => "area",
        CadParameterPhysicalType::Volume => "volume",
        CadParameterPhysicalType::Mass => "mass",
        CadParameterPhysicalType::Temperature => "temperature",
    }
}

fn fea_type_name(value: &FeaType) -> &'static str {
    match value {
        FeaType::Integer => "integer",
        FeaType::Real => "real",
        FeaType::Quantity(_) => "a physically incompatible quantity",
        _ => "a non-scalar value",
    }
}

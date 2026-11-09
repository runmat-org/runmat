use super::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaType, PhysicalDimension,
};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey,
    FeaSymbolKind, FeaSymbolOrigin, LogicalSourcePath,
};
use runmat_analysis_core::AuthoredRegionSource;
use runmat_geometry_core::GeometryDefinitionRef;
use runmat_geometry_core::GeometryRegionRole;
use runmat_meshing_core::{
    ElementOrder, MeshDefinition, MeshDimension, MeshSizeControl, MeshSizeControlTarget,
    MeshingRequestSettings,
};
use std::collections::{BTreeMap, BTreeSet};

pub fn lower_mesh_definitions(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    geometries: &BTreeMap<
        (LogicalSourcePath, FeaSymbolKey),
        runmat_geometry_core::GeometryDefinition,
    >,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), MeshDefinition> {
    let mut result = BTreeMap::new();
    for (identity, declaration) in declarations {
        if identity.1.kind != FeaSymbolKind::Mesh {
            continue;
        }
        let initial_errors = diagnostics.len();
        let definition = definition(bound, declaration, geometries, diagnostics);
        if diagnostics.len() != initial_errors {
            continue;
        }
        let Some(definition) = definition else {
            continue;
        };
        match definition.validate() {
            Ok(()) => {
                result.insert(identity.clone(), definition);
            }
            Err(error) => diagnostic(
                declaration,
                declaration.span,
                error.to_string(),
                diagnostics,
            ),
        }
    }
    result
}

fn definition(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    geometries: &BTreeMap<
        (LogicalSourcePath, FeaSymbolKey),
        runmat_geometry_core::GeometryDefinition,
    >,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<MeshDefinition> {
    let geometry = geometry_reference(bound, declaration, diagnostics)?;
    let section = geometries.values().any(|definition| {
        definition.module_path == geometry.module_path
            && definition.geometry_id == geometry.geometry_id
            && matches!(
                definition.source,
                runmat_geometry_core::GeometryDefinitionSource::CrossSection(_)
            )
    });
    if section {
        return cross_section_definition(declaration, geometry, diagnostics);
    }
    let (block, dimension) = element_block(declaration, diagnostics)?;
    let (header, settings_block) = block;
    reject_unsupported_top_level(declaration, diagnostics);
    reject_unsupported_settings(declaration, header, settings_block, diagnostics);

    // Omitted settings use the canonical mesher defaults; explicit settings still pass the
    // normal type, duplicate, and value checks. Shell triangles and beam segments are linear;
    // solids choose their tetrahedral order.
    let defaults = MeshingRequestSettings::default();
    let order = if dimension != MeshDimension::Solid {
        Some(ElementOrder::Tet4)
    } else if assignment(settings_block, "order").is_none() {
        Some(defaults.element_order)
    } else {
        required_enum(declaration, settings_block, "order", diagnostics).and_then(|value| {
            match value {
                "linear" => Some(ElementOrder::Tet4),
                "quadratic" => Some(ElementOrder::Tet10),
                _ => {
                    diagnostic(
                        declaration,
                        declaration.span,
                        "mesh `order` must be `linear` or `quadratic`",
                        diagnostics,
                    );
                    None
                }
            }
        })
    };
    let target_edge_length_m = if assignment(settings_block, "target_size").is_none() {
        Some(defaults.target_edge_length_m.canonical_value())
    } else {
        required_length(declaration, settings_block, "target_size", diagnostics)
    };
    let maximum_chordal_deviation_m = if assignment(settings_block, "maximum_deviation").is_none() {
        Some(defaults.maximum_chordal_deviation_m.canonical_value())
    } else {
        required_length(
            declaration,
            settings_block,
            "maximum_deviation",
            diagnostics,
        )
    };

    let maximum_grading_ratio = optional_scalar(
        declaration,
        settings_block,
        "maximum_grading_ratio",
        diagnostics,
    )
    .unwrap_or(defaults.maximum_grading_ratio);
    let deterministic_seed = optional_integer(
        declaration,
        settings_block,
        "deterministic_seed",
        diagnostics,
    )
    .unwrap_or(defaults.deterministic_seed);
    let cells_across_gap =
        optional_integer(declaration, settings_block, "cells_across_gap", diagnostics).and_then(
            |cells| match u32::try_from(cells) {
                Ok(cells @ 1..=64) => Some(cells),
                _ => {
                    coded(
                        declaration,
                        declaration.span,
                        "fea.semantic.mesh.cells_across_gap",
                        "mesh `cells_across_gap` must be an integer from 1 to 64",
                        diagnostics,
                    );
                    None
                }
            },
        );
    let size_controls = size_controls(
        bound,
        declaration,
        &geometry,
        settings_block,
        target_edge_length_m,
        diagnostics,
    );
    let settings = MeshingRequestSettings {
        element_order: order?,
        dimension,
        deterministic_seed,
        target_edge_length_m: runmat_types::physical::Length::new(target_edge_length_m?),
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(
            maximum_chordal_deviation_m?,
        ),
        maximum_grading_ratio,
        cells_across_gap,
        size_controls,
        ..defaults
    };

    Some(MeshDefinition {
        module_path: declaration.module.as_str().to_string(),
        mesh_id: declaration.key.name.clone(),
        geometry,
        settings,
    })
}

/// `mesh <name> on geometry.<section> { triangles { edge_size, growth, max_size } }`: the
/// graded triangles of a board cross-section, meshed by the transmission-line solver.
fn cross_section_definition(
    declaration: &CheckedFeaDeclaration,
    geometry: GeometryDefinitionRef,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<MeshDefinition> {
    let blocks = declaration
        .statements
        .iter()
        .map(|statement| match &statement.kind {
            CheckedFeaStatementKind::Block {
                header, statements, ..
            } if header.trim() == "triangles" => Some(statements.as_slice()),
            _ => {
                diagnostic(
                    declaration,
                    statement.span,
                    "a mesh on a cross_section accepts only its `triangles { edge_size, growth, max_size }` block",
                    diagnostics,
                );
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;
    let statements: &[CheckedFeaStatement] = match blocks.as_slice() {
        [] => &[],
        [block] => block,
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                "mesh accepts one `triangles` block",
                diagnostics,
            );
            return None;
        }
    };
    let errors = diagnostics.len();
    let mut seen = BTreeSet::new();
    for statement in statements {
        match &statement.kind {
            CheckedFeaStatementKind::Assignment { target, .. }
                if target.len() == 1
                    && ["edge_size", "growth", "max_size"].contains(&target[0].as_str()) =>
            {
                if !seen.insert(target[0].as_str()) {
                    diagnostic(
                        declaration,
                        statement.span,
                        format!("duplicate triangles setting `{}`", target[0]),
                        diagnostics,
                    );
                }
            }
            _ => diagnostic(
                declaration,
                statement.span,
                "cross-section `triangles` accepts edge_size, growth and max_size",
                diagnostics,
            ),
        }
    }
    let length = |name: &str, diagnostics: &mut Vec<FeaDiagnostic>| {
        assignment(statements, name)
            .is_some()
            .then(|| required_length(declaration, statements, name, diagnostics))
            .flatten()
    };
    let section = runmat_meshing_core::CrossSectionMeshSettings {
        edge_size_m: length("edge_size", diagnostics),
        growth: optional_scalar(declaration, statements, "growth", diagnostics),
        max_size_m: length("max_size", diagnostics),
    };
    if diagnostics.len() != errors {
        return None;
    }
    Some(MeshDefinition {
        module_path: declaration.module.as_str().to_string(),
        mesh_id: declaration.key.name.clone(),
        geometry,
        settings: MeshingRequestSettings {
            dimension: MeshDimension::Surface,
            cross_section: Some(section),
            ..MeshingRequestSettings::default()
        },
    })
}

fn geometry_reference(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryDefinitionRef> {
    let Some([namespace, name]) = declaration.target.as_deref() else {
        diagnostic(
            declaration,
            declaration.span,
            "mesh requires `on geometry.<name>`",
            diagnostics,
        );
        return None;
    };
    if namespace != "geometry" {
        diagnostic(
            declaration,
            declaration.span,
            "mesh target must use the `geometry` namespace",
            diagnostics,
        );
        return None;
    }
    let key = FeaSymbolKey {
        kind: FeaSymbolKind::Geometry,
        name: name.clone(),
    };
    let Some(symbol) = bound
        .symbols
        .get(&declaration.module)
        .and_then(|symbols| symbols.get(&key))
    else {
        diagnostic(
            declaration,
            declaration.span,
            format!("mesh references unknown geometry `{name}`"),
            diagnostics,
        );
        return None;
    };
    match &symbol.origin {
        FeaSymbolOrigin::Local => Some(GeometryDefinitionRef {
            module_path: declaration.module.as_str().to_string(),
            geometry_id: name.clone(),
        }),
        FeaSymbolOrigin::Import {
            target_module,
            target,
        } => Some(GeometryDefinitionRef {
            module_path: target_module.as_str().to_string(),
            geometry_id: target.name.clone(),
        }),
        FeaSymbolOrigin::Function => {
            diagnostic(
                declaration,
                declaration.span,
                "mesh target resolved to a function",
                diagnostics,
            );
            None
        }
    }
}

/// The element block names the solver elements: `tetrahedra` for solids, `triangles` for
/// sheet faces meshed as shells (with their edges as segments), `lines` for edges only.
const ELEMENT_BLOCKS: [(&str, MeshDimension); 3] = [
    ("tetrahedra", MeshDimension::Solid),
    ("triangles", MeshDimension::Surface),
    ("lines", MeshDimension::Curve),
];

type ElementBlock<'a> = ((&'static str, &'a [CheckedFeaStatement]), MeshDimension);

fn element_block<'a>(
    declaration: &'a CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ElementBlock<'a>> {
    let matching = declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::Block {
                header, statements, ..
            } = &statement.kind
            else {
                return None;
            };
            ELEMENT_BLOCKS
                .iter()
                .find(|(name, _)| header.trim() == *name)
                .map(|(name, dimension)| ((*name, statements.as_slice()), *dimension))
        })
        .collect::<Vec<_>>();
    if matching.is_empty() {
        // An omitted element block requests solid tetrahedra with the canonical mesher defaults.
        Some(((ELEMENT_BLOCKS[0].0, &[]), ELEMENT_BLOCKS[0].1))
    } else if matching.len() != 1 {
        diagnostic(
            declaration,
            declaration.span,
            "mesh accepts at most one `tetrahedra`, `triangles`, or `lines` block",
            diagnostics,
        );
        None
    } else {
        matching.into_iter().next()
    }
}

fn reject_unsupported_top_level(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    for statement in &declaration.statements {
        if matches!(
            &statement.kind,
            CheckedFeaStatementKind::Block { header, .. }
                if ELEMENT_BLOCKS.iter().any(|(name, _)| header.trim() == *name)
        ) {
            continue;
        }
        diagnostic(
            declaration,
            statement.span,
            "mesh accepts only its `tetrahedra`, `triangles`, or `lines` request block",
            diagnostics,
        );
    }
}

fn reject_unsupported_settings(
    declaration: &CheckedFeaDeclaration,
    block: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let mut supported = BTreeSet::from([
        "target_size",
        "maximum_deviation",
        "maximum_grading_ratio",
        "deterministic_seed",
    ]);
    if block == "tetrahedra" {
        supported.insert("order");
    }
    if block == "tetrahedra" {
        supported.insert("cells_across_gap");
    }
    let mut seen = BTreeSet::new();
    for statement in statements {
        if matches!(&statement.kind, CheckedFeaStatementKind::Block { header, .. }
            if block == "tetrahedra" && refine_name(header).is_some())
        {
            continue;
        }
        let CheckedFeaStatementKind::Assignment {
            target, relation, ..
        } = &statement.kind
        else {
            diagnostic(
                declaration,
                statement.span,
                format!("{block} settings must be assignments"),
                diagnostics,
            );
            continue;
        };
        let Some(name) = target.as_slice().first().filter(|_| target.len() == 1) else {
            diagnostic(
                declaration,
                statement.span,
                "mesh setting names cannot be qualified",
                diagnostics,
            );
            continue;
        };
        if *relation != runmat_design_syntax::FeaAssignmentRelation::Equal {
            diagnostic(
                declaration,
                statement.span,
                "mesh settings require `=`",
                diagnostics,
            );
        }
        if !supported.contains(name.as_str()) {
            diagnostic(
                declaration,
                statement.span,
                format!("unsupported {block} setting `{name}`"),
                diagnostics,
            );
        } else if !seen.insert(name.as_str()) {
            diagnostic(
                declaration,
                statement.span,
                format!("duplicate {block} setting `{name}`"),
                diagnostics,
            );
        }
    }
}

/// `refine <name> { on = geometry.<g>.region.<r>  target_size = <length> }` blocks of a
/// `tetrahedra` request: one local size control per named solid, face or edge region.
fn refine_name(header: &str) -> Option<&str> {
    let mut words = header.split_whitespace();
    match (words.next(), words.next(), words.next()) {
        (Some("refine"), Some(name), None) => Some(name),
        _ => None,
    }
}

fn size_controls(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    geometry: &GeometryDefinitionRef,
    statements: &[CheckedFeaStatement],
    global_size_m: Option<f64>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<MeshSizeControl> {
    let mut controls = BTreeMap::<String, MeshSizeControl>::new();
    let mut regions = BTreeMap::<String, String>::new();
    for statement in statements {
        let CheckedFeaStatementKind::Block {
            header,
            statements: body,
            ..
        } = &statement.kind
        else {
            continue;
        };
        let Some(name) = refine_name(header) else {
            continue;
        };
        let errors = diagnostics.len();
        for member in body {
            match &member.kind {
                CheckedFeaStatementKind::Assignment { target, .. }
                    if target.len() == 1 && ["on", "target_size"].contains(&target[0].as_str()) => {
                }
                _ => coded(
                    declaration,
                    member.span,
                    "fea.semantic.mesh.control_setting",
                    format!("refine `{name}` accepts only `on` and `target_size`"),
                    diagnostics,
                ),
            }
        }
        let Some(on) = assignment(body, "on") else {
            missing(declaration, &format!("on` in `refine {name}"), diagnostics);
            continue;
        };
        let target = match super::model::region_source(bound, declaration, on, diagnostics) {
            Some(AuthoredRegionSource::Geometry(region)) => {
                if region.module_path != geometry.module_path
                    || region.geometry_id != geometry.geometry_id
                {
                    coded(
                        declaration,
                        on.span,
                        "fea.semantic.mesh.control_region",
                        format!(
                            "refine `{name}` selects a region of geometry `{}`, not the meshed geometry `{}`",
                            region.geometry_id, geometry.geometry_id
                        ),
                        diagnostics,
                    );
                    None
                } else {
                    match region.role {
                        GeometryRegionRole::Solid => {
                            Some((region.region_id, MeshSizeControlTarget::Solid))
                        }
                        GeometryRegionRole::Face => {
                            Some((region.region_id, MeshSizeControlTarget::Face))
                        }
                        GeometryRegionRole::Edge => {
                            Some((region.region_id, MeshSizeControlTarget::Edge))
                        }
                        role => {
                            coded(
                                declaration,
                                on.span,
                                "fea.semantic.mesh.control_region",
                                format!("refine `{name}` requires a solid, face or edge region, not a {role:?} region"),
                                diagnostics,
                            );
                            None
                        }
                    }
                }
            }
            Some(AuthoredRegionSource::ModelInput(_)) => {
                coded(
                    declaration,
                    on.span,
                    "fea.semantic.mesh.control_region",
                    format!("refine `{name}` requires a geometry region, not an input"),
                    diagnostics,
                );
                None
            }
            None => None,
        };
        let size = required_length(declaration, body, "target_size", diagnostics);
        let (Some((region_id, target)), Some(size)) = (target, size) else {
            continue;
        };
        if diagnostics.len() != errors {
            continue;
        }
        if !size.is_finite() || size <= 0.0 {
            coded(
                declaration,
                statement.span,
                "fea.semantic.mesh.control_size",
                format!("refine `{name}` target_size must be positive"),
                diagnostics,
            );
            continue;
        }
        if global_size_m.is_some_and(|global| size >= global) {
            coded(
                declaration,
                statement.span,
                "fea.semantic.mesh.control_size",
                format!(
                    "refine `{name}` target_size must be smaller than the mesh target_size; \
                     local controls refine, and the smallest requested size wins where controls meet"
                ),
                diagnostics,
            );
            continue;
        }
        if let Some(previous) = regions.insert(region_id.clone(), name.to_string()) {
            coded(
                declaration,
                statement.span,
                "fea.semantic.mesh.control_conflict",
                format!(
                    "refine `{name}` and refine `{previous}` both control region `{region_id}`"
                ),
                diagnostics,
            );
            continue;
        }
        if controls.contains_key(name) {
            coded(
                declaration,
                statement.span,
                "fea.semantic.mesh.control_conflict",
                format!("duplicate refine `{name}`"),
                diagnostics,
            );
            continue;
        }
        controls.insert(
            name.to_string(),
            MeshSizeControl {
                control_id: name.to_string(),
                region_id,
                target,
                target_size_m: size,
            },
        );
    }
    controls.into_values().collect()
}

fn assignment<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a CheckedExpression> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        (target.as_slice() == [name]).then_some(value)
    })
}

fn required_enum<'a>(
    declaration: &CheckedFeaDeclaration,
    statements: &'a [CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a str> {
    let Some(value) = assignment(statements, name) else {
        missing(declaration, name, diagnostics);
        return None;
    };
    let CheckedExpressionKind::Enum(value) = &value.kind else {
        invalid(declaration, value, name, "enum value", diagnostics);
        return None;
    };
    Some(value)
}

fn required_length(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let Some(value) = assignment(statements, name) else {
        missing(declaration, name, diagnostics);
        return None;
    };
    if !(value.value_type).is_quantity((PhysicalDimension::LENGTH).into()) {
        if let Some((message, label)) =
            super::units::unit_mismatch(name, value, PhysicalDimension::LENGTH)
        {
            diagnostics.push(
                FeaDiagnostic::error(
                    "fea.semantic.mesh",
                    message,
                    crate::FeaDiagnosticPhase::Lowering,
                    Some(crate::FeaDiagnosticLocation {
                        path: declaration.module.clone(),
                        span: value.span,
                    }),
                )
                .with_label(label),
            );
            return None;
        }
        invalid(declaration, value, name, "length quantity", diagnostics);
        return None;
    }
    scalar(value).or_else(|| {
        invalid(
            declaration,
            value,
            name,
            "constant length quantity",
            diagnostics,
        );
        None
    })
}

fn optional_scalar(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let value = assignment(statements, name)?;
    if !matches!(value.value_type, FeaType::Integer | FeaType::Real) {
        invalid(
            declaration,
            value,
            name,
            "dimensionless scalar",
            diagnostics,
        );
        return None;
    }
    scalar(value).or_else(|| {
        invalid(declaration, value, name, "constant scalar", diagnostics);
        None
    })
}

fn optional_integer(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<u64> {
    let value = assignment(statements, name)?;
    if value.value_type != FeaType::Integer {
        invalid(
            declaration,
            value,
            name,
            "non-negative integer",
            diagnostics,
        );
        return None;
    }
    let CheckedExpressionKind::Number(source) = &value.kind else {
        invalid(
            declaration,
            value,
            name,
            "constant non-negative integer",
            diagnostics,
        );
        return None;
    };
    source
        .parse()
        .ok()
        .and_then(runmat_types::wire_integer::bounded)
        .or_else(|| {
            invalid(
                declaration,
                value,
                name,
                "non-negative integer of at most 2^53 - 1",
                diagnostics,
            );
            None
        })
}

fn scalar(value: &CheckedExpression) -> Option<f64> {
    match &value.kind {
        CheckedExpressionKind::Number(source) => source.parse().ok(),
        CheckedExpressionKind::CadParameterValue(source) => Some(source.value_si()),
        CheckedExpressionKind::Quantity { value, unit, .. } => {
            unit.to_canonical(scalar(value)?).ok()
        }
        CheckedExpressionKind::Unary { operator, operand } => {
            let scalar = scalar(operand)?;
            Some(match operator {
                runmat_design_syntax::FeaUnaryOperator::Positive => scalar,
                runmat_design_syntax::FeaUnaryOperator::Negative => -scalar,
            })
        }
        _ => None,
    }
}

fn missing(declaration: &CheckedFeaDeclaration, name: &str, diagnostics: &mut Vec<FeaDiagnostic>) {
    diagnostic(
        declaration,
        declaration.span,
        format!("mesh element block requires `{name}`"),
        diagnostics,
    );
}

fn invalid(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    name: &str,
    expected: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostic(
        declaration,
        value.span,
        format!("mesh setting `{name}` requires a {expected}"),
        diagnostics,
    );
}

fn coded(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    code: &'static str,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        code,
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}

fn diagnostic(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.mesh",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}

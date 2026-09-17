use super::{diagnostic, material_source, property_region_source, property_target};
use crate::{
    BoundFeaProgram, CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind,
    FeaDiagnostic, FeaType, PhysicalDimension,
};
use runmat_analysis_core::{
    AuthoredLayupDefinition, AuthoredLayupSource, AuthoredOrientationDefinition,
    AuthoredOrientationSource, AuthoredPly, AuthoredPropertyAssignment,
    AuthoredPropertyDefinitionRef, AuthoredSectionDefinition, AuthoredSectionSource,
    AxisOrientation, CartesianOrientation, CircularBeamSection, GeneralBeamSection,
    HomogeneousShellSection, RectangularBeamSection, TubularBeamSection,
};
use runmat_parser::fea::FeaPropertyDefinitionKind;

mod values;
use values::*;

pub(super) struct LoweredProperties {
    pub sections: Vec<AuthoredSectionDefinition>,
    pub layups: Vec<AuthoredLayupDefinition>,
    pub orientations: Vec<AuthoredOrientationDefinition>,
    pub assignments: Vec<AuthoredPropertyAssignment>,
}

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> LoweredProperties {
    let mut sections = Vec::new();
    let mut layups = Vec::new();
    let mut orientations = Vec::new();
    for statement in &declaration.statements {
        let CheckedFeaStatementKind::PropertyDefinition {
            kind,
            name,
            statements,
        } = &statement.kind
        else {
            continue;
        };
        match kind {
            FeaPropertyDefinitionKind::Section => {
                if let Some(value) = section(declaration, name, statements, diagnostics) {
                    sections.push(value);
                }
            }
            FeaPropertyDefinitionKind::Layup => {
                if let Some(value) = layup(declaration, bound, name, statements, diagnostics) {
                    layups.push(value);
                }
            }
            FeaPropertyDefinitionKind::Orientation => {
                if let Some(value) = orientation(declaration, name, statements, diagnostics) {
                    orientations.push(value);
                }
            }
        }
    }
    let assignments = declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::PropertyAssignment { property, target } = &statement.kind
            else {
                return None;
            };
            match property.value_type {
                FeaType::Section => Some(AuthoredPropertyAssignment::Section {
                    section: AuthoredSectionSource::Definition(property_reference(
                        declaration,
                        reference_name(declaration, property, "section", diagnostics)?,
                    )),
                    target: property_target(bound, declaration, target, diagnostics)?,
                }),
                FeaType::Layup => Some(AuthoredPropertyAssignment::Layup {
                    layup: AuthoredLayupSource::Definition(property_reference(
                        declaration,
                        reference_name(declaration, property, "layup", diagnostics)?,
                    )),
                    target: property_region_source(bound, declaration, target, diagnostics)?,
                }),
                FeaType::Orientation => Some(AuthoredPropertyAssignment::Orientation {
                    orientation: AuthoredOrientationSource::Definition(property_reference(
                        declaration,
                        reference_name(declaration, property, "orientation", diagnostics)?,
                    )),
                    target: property_target(bound, declaration, target, diagnostics)?,
                }),
                FeaType::Declaration(_) | FeaType::Error => None,
                _ => {
                    diagnostic(
                        declaration,
                        property.span,
                        "assignment source must be a material, section, layup, or orientation",
                        diagnostics,
                    );
                    None
                }
            }
        })
        .collect();
    LoweredProperties {
        sections,
        layups,
        orientations,
        assignments,
    }
}

fn section(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredSectionDefinition> {
    let (kind, body) = sole_kind_block(declaration, statements, "section", diagnostics)?;
    let supported = match kind {
        "general_beam" => &[
            "area",
            "iy",
            "iz",
            "torsion_constant",
            "outer_fiber_y",
            "outer_fiber_z",
            "torsion_outer_radius",
        ][..],
        "rectangular_beam" => &["width", "height"][..],
        "circular_beam" => &["radius"][..],
        "tubular_beam" => &["outer_radius", "inner_radius"][..],
        "homogeneous_shell" => &[
            "thickness",
            "offset_fraction",
            "shear_correction",
            "drilling_stiffness_scale",
            "integration_points",
        ][..],
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported section formulation `{kind}`"),
                diagnostics,
            );
            return None;
        }
    };
    validate_assignments(declaration, body, supported, diagnostics);
    Some(match kind {
        "general_beam" => AuthoredSectionDefinition::GeneralBeam(GeneralBeamSection {
            section_id: id.into(),
            area_m2: required_quantity(
                declaration,
                body,
                "area",
                PhysicalDimension::AREA,
                diagnostics,
            )?,
            iy_m4: required_fourth_length(declaration, body, "iy", diagnostics)?,
            iz_m4: required_fourth_length(declaration, body, "iz", diagnostics)?,
            torsion_j_m4: required_fourth_length(
                declaration,
                body,
                "torsion_constant",
                diagnostics,
            )?,
            outer_fiber_y_m: required_length(declaration, body, "outer_fiber_y", diagnostics)?,
            outer_fiber_z_m: required_length(declaration, body, "outer_fiber_z", diagnostics)?,
            torsion_outer_radius_m: required_length(
                declaration,
                body,
                "torsion_outer_radius",
                diagnostics,
            )?,
        }),
        "rectangular_beam" => AuthoredSectionDefinition::RectangularBeam(RectangularBeamSection {
            section_id: id.into(),
            width_m: required_length(declaration, body, "width", diagnostics)?,
            height_m: required_length(declaration, body, "height", diagnostics)?,
        }),
        "circular_beam" => AuthoredSectionDefinition::CircularBeam(CircularBeamSection {
            section_id: id.into(),
            radius_m: required_length(declaration, body, "radius", diagnostics)?,
        }),
        "tubular_beam" => AuthoredSectionDefinition::TubularBeam(TubularBeamSection {
            section_id: id.into(),
            outer_radius_m: required_length(declaration, body, "outer_radius", diagnostics)?,
            inner_radius_m: required_length(declaration, body, "inner_radius", diagnostics)?,
        }),
        "homogeneous_shell" => {
            AuthoredSectionDefinition::HomogeneousShell(HomogeneousShellSection {
                section_id: id.into(),
                thickness_m: required_length(declaration, body, "thickness", diagnostics)?,
                offset_fraction: required_scalar(
                    declaration,
                    body,
                    "offset_fraction",
                    diagnostics,
                )?,
                shear_correction: required_scalar(
                    declaration,
                    body,
                    "shear_correction",
                    diagnostics,
                )?,
                drilling_stiffness_scale: required_scalar(
                    declaration,
                    body,
                    "drilling_stiffness_scale",
                    diagnostics,
                )?,
                integration_points: required_u32(
                    declaration,
                    body,
                    "integration_points",
                    diagnostics,
                )?,
            })
        }
        _ => return None,
    })
}

fn orientation(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredOrientationDefinition> {
    let (kind, body) = sole_kind_block(declaration, statements, "orientation", diagnostics)?;
    Some(match kind {
        "global_cartesian" => {
            validate_assignments(declaration, body, &[], diagnostics);
            AuthoredOrientationDefinition::GlobalCartesian {
                orientation_id: id.into(),
            }
        }
        "cartesian" => {
            validate_assignments(
                declaration,
                body,
                &["origin", "x_axis", "y_axis"],
                diagnostics,
            );
            AuthoredOrientationDefinition::Cartesian(CartesianOrientation {
                orientation_id: id.into(),
                origin_m: required_vector(
                    declaration,
                    body,
                    "origin",
                    PhysicalDimension::LENGTH,
                    diagnostics,
                )?,
                x_axis: required_vector(
                    declaration,
                    body,
                    "x_axis",
                    PhysicalDimension::DIMENSIONLESS,
                    diagnostics,
                )?,
                y_axis: required_vector(
                    declaration,
                    body,
                    "y_axis",
                    PhysicalDimension::DIMENSIONLESS,
                    diagnostics,
                )?,
            })
        }
        "cylindrical" | "spherical" => {
            validate_assignments(
                declaration,
                body,
                &["origin", "axis", "reference_direction"],
                diagnostics,
            );
            let value = AxisOrientation {
                orientation_id: id.into(),
                origin_m: required_vector(
                    declaration,
                    body,
                    "origin",
                    PhysicalDimension::LENGTH,
                    diagnostics,
                )?,
                axis: required_vector(
                    declaration,
                    body,
                    "axis",
                    PhysicalDimension::DIMENSIONLESS,
                    diagnostics,
                )?,
                reference_direction: required_vector(
                    declaration,
                    body,
                    "reference_direction",
                    PhysicalDimension::DIMENSIONLESS,
                    diagnostics,
                )?,
            };
            if kind == "cylindrical" {
                AuthoredOrientationDefinition::Cylindrical(value)
            } else {
                AuthoredOrientationDefinition::Spherical(value)
            }
        }
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported orientation frame `{kind}`"),
                diagnostics,
            );
            return None;
        }
    })
}

fn layup(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    id: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredLayupDefinition> {
    validate_layup_members(declaration, statements, diagnostics);
    let reference = required_assignment(
        declaration,
        statements,
        "reference_orientation",
        diagnostics,
    )?;
    let reference_orientation = property_reference(
        declaration,
        reference_name(declaration, reference, "orientation", diagnostics)?,
    );
    let stacking = required_i8(declaration, statements, "stacking_direction", diagnostics)?;
    let mut plies = Vec::new();
    for statement in statements {
        let CheckedFeaStatementKind::Block {
            header,
            statements: body,
            ..
        } = &statement.kind
        else {
            continue;
        };
        let Some(ply_id) = header.trim().strip_prefix("ply ") else {
            diagnostic(
                declaration,
                statement.span,
                "layup nested blocks must use `ply <name>`",
                diagnostics,
            );
            continue;
        };
        validate_assignments(
            declaration,
            body,
            &["material", "thickness", "angle", "integration_points"],
            diagnostics,
        );
        let material = required_assignment(declaration, body, "material", diagnostics)?;
        plies.push(AuthoredPly {
            ply_id: ply_id.trim().into(),
            material: material_source(bound, declaration, material, diagnostics)?,
            thickness_m: required_length(declaration, body, "thickness", diagnostics)?,
            angle_rad: required_quantity(
                declaration,
                body,
                "angle",
                PhysicalDimension::DIMENSIONLESS,
                diagnostics,
            )?,
            integration_points: required_u32(declaration, body, "integration_points", diagnostics)?,
        });
    }
    Some(AuthoredLayupDefinition {
        layup_id: id.into(),
        reference_orientation,
        stacking_direction: stacking,
        plies,
    })
}

fn validate_layup_members(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let mut assignments = std::collections::BTreeSet::new();
    let mut plies = std::collections::BTreeSet::new();
    for statement in statements {
        match &statement.kind {
            CheckedFeaStatementKind::Assignment {
                target, relation, ..
            } => {
                let Some(name) = target.first().filter(|_| target.len() == 1) else {
                    diagnostic(
                        declaration,
                        statement.span,
                        "layup member names cannot be qualified",
                        diagnostics,
                    );
                    continue;
                };
                if *relation != runmat_parser::fea::FeaAssignmentRelation::Equal
                    || !matches!(
                        name.as_str(),
                        "reference_orientation" | "stacking_direction"
                    )
                    || !assignments.insert(name)
                {
                    diagnostic(
                        declaration,
                        statement.span,
                        format!("invalid or duplicate layup member `{name}`"),
                        diagnostics,
                    );
                }
            }
            CheckedFeaStatementKind::Block { header, .. } => {
                let Some(name) = header.trim().strip_prefix("ply ") else {
                    diagnostic(
                        declaration,
                        statement.span,
                        "layup nested blocks must use `ply <name>`",
                        diagnostics,
                    );
                    continue;
                };
                if name.trim().is_empty() || !plies.insert(name.trim()) {
                    diagnostic(
                        declaration,
                        statement.span,
                        "layup ply names must be nonempty and unique",
                        diagnostics,
                    );
                }
            }
            _ => diagnostic(
                declaration,
                statement.span,
                "layup accepts only its two settings and ordered ply blocks",
                diagnostics,
            ),
        }
    }
}

fn property_reference(
    declaration: &CheckedFeaDeclaration,
    property_id: String,
) -> AuthoredPropertyDefinitionRef {
    AuthoredPropertyDefinitionRef {
        module_path: declaration.module.as_str().to_string(),
        model_id: declaration.key.name.clone(),
        property_id,
    }
}

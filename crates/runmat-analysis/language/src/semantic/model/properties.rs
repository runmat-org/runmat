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
    HomogeneousShellReferenceSurface, HomogeneousShellSection, PlyOrderDirection,
    RectangularBeamSection, ShellReferenceSurface, TubularBeamSection,
};
use runmat_design_syntax::FeaModelDefinitionKind;

mod values;
pub(super) use values::*;

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
        let CheckedFeaStatementKind::ModelDefinition {
            kind,
            name,
            statements,
        } = &statement.kind
        else {
            continue;
        };
        match kind {
            FeaModelDefinitionKind::Section => {
                if let Some(value) = section(declaration, name, statements, diagnostics) {
                    sections.push(value);
                }
            }
            FeaModelDefinitionKind::Layup => {
                if let Some(value) = layup(declaration, bound, name, statements, diagnostics) {
                    layups.push(value);
                }
            }
            FeaModelDefinitionKind::Orientation => {
                if let Some(value) = orientation(declaration, name, statements, diagnostics) {
                    orientations.push(value);
                }
            }
            _ => {}
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
            "reference_surface",
            "reference_offset",
            "shear_correction",
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
        "general_beam" => {
            AuthoredSectionDefinition::GeneralBeam(GeneralBeamSection {
                section_id: id.into(),
                area_m2: runmat_types::physical::Area::new(required_quantity(
                    declaration,
                    body,
                    "area",
                    PhysicalDimension::AREA,
                    diagnostics,
                )?),
                iy_m4: runmat_types::physical::SecondMomentOfArea::new(required_fourth_length(
                    declaration,
                    body,
                    "iy",
                    diagnostics,
                )?),
                iz_m4: runmat_types::physical::SecondMomentOfArea::new(required_fourth_length(
                    declaration,
                    body,
                    "iz",
                    diagnostics,
                )?),
                torsion_j_m4: runmat_types::physical::SecondMomentOfArea::new(
                    required_fourth_length(declaration, body, "torsion_constant", diagnostics)?,
                ),
                outer_fiber_y_m: runmat_types::physical::Length::new(required_length(
                    declaration,
                    body,
                    "outer_fiber_y",
                    diagnostics,
                )?),
                outer_fiber_z_m: runmat_types::physical::Length::new(required_length(
                    declaration,
                    body,
                    "outer_fiber_z",
                    diagnostics,
                )?),
                torsion_outer_radius_m: runmat_types::physical::Length::new(required_length(
                    declaration,
                    body,
                    "torsion_outer_radius",
                    diagnostics,
                )?),
            })
        }
        "rectangular_beam" => AuthoredSectionDefinition::RectangularBeam(RectangularBeamSection {
            section_id: id.into(),
            width_m: runmat_types::physical::Length::new(required_length(
                declaration,
                body,
                "width",
                diagnostics,
            )?),
            height_m: runmat_types::physical::Length::new(required_length(
                declaration,
                body,
                "height",
                diagnostics,
            )?),
        }),
        "circular_beam" => AuthoredSectionDefinition::CircularBeam(CircularBeamSection {
            section_id: id.into(),
            radius_m: runmat_types::physical::Length::new(required_length(
                declaration,
                body,
                "radius",
                diagnostics,
            )?),
        }),
        "tubular_beam" => AuthoredSectionDefinition::TubularBeam(TubularBeamSection {
            section_id: id.into(),
            outer_radius_m: runmat_types::physical::Length::new(required_length(
                declaration,
                body,
                "outer_radius",
                diagnostics,
            )?),
            inner_radius_m: runmat_types::physical::Length::new(required_length(
                declaration,
                body,
                "inner_radius",
                diagnostics,
            )?),
        }),
        "homogeneous_shell" => {
            AuthoredSectionDefinition::HomogeneousShell(HomogeneousShellSection {
                section_id: id.into(),
                thickness_m: runmat_types::physical::Length::new(required_length(
                    declaration,
                    body,
                    "thickness",
                    diagnostics,
                )?),
                reference_surface: homogeneous_reference_surface(declaration, body, diagnostics)?,
                shear_correction: required_scalar(
                    declaration,
                    body,
                    "shear_correction",
                    diagnostics,
                )?,
            })
        }
        _ => return None,
    })
}

fn homogeneous_reference_surface(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<HomogeneousShellReferenceSurface> {
    let surface = match required_enum(declaration, statements, "reference_surface", diagnostics)? {
        "mid_surface" => HomogeneousShellReferenceSurface::MidSurface,
        "negative_normal_surface" => HomogeneousShellReferenceSurface::NegativeNormalSurface,
        "positive_normal_surface" => HomogeneousShellReferenceSurface::PositiveNormalSurface,
        "offset_from_mid_surface" => HomogeneousShellReferenceSurface::OffsetFromMidSurface(
            optional_property_quantity(
                declaration,
                statements,
                "reference_offset",
                PhysicalDimension::LENGTH,
                diagnostics,
            )?
            .or_else(|| {
                diagnostic(
                    declaration,
                    declaration.span,
                    "offset reference surface requires `reference_offset`",
                    diagnostics,
                );
                None
            })?,
        ),
        value => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported homogeneous shell reference surface `{value}`"),
                diagnostics,
            );
            return None;
        }
    };
    if !matches!(
        surface,
        HomogeneousShellReferenceSurface::OffsetFromMidSurface(_)
    ) && optional_property_assignment(statements, "reference_offset").is_some()
    {
        diagnostic(
            declaration,
            declaration.span,
            "`reference_offset` is valid only with `offset_from_mid_surface`",
            diagnostics,
        );
        return None;
    }
    Some(surface)
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
                origin_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(required_vector(
                    declaration,
                    body,
                    "origin",
                    PhysicalDimension::LENGTH,
                    diagnostics,
                )?),
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
                origin_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(required_vector(
                    declaration,
                    body,
                    "origin",
                    PhysicalDimension::LENGTH,
                    diagnostics,
                )?),
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
    let ply_order = match required_enum(declaration, statements, "ply_order", diagnostics)? {
        "along_surface_normal" => PlyOrderDirection::AlongSurfaceNormal,
        "opposite_surface_normal" => PlyOrderDirection::OppositeSurfaceNormal,
        value => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported ply order `{value}`"),
                diagnostics,
            );
            return None;
        }
    };
    let reference_surface =
        match required_enum(declaration, statements, "reference_surface", diagnostics)? {
            "mid_surface" => ShellReferenceSurface::MidSurface,
            "first_ply_surface" => ShellReferenceSurface::FirstPlySurface,
            "last_ply_surface" => ShellReferenceSurface::LastPlySurface,
            "offset_from_mid_surface" => ShellReferenceSurface::OffsetFromMidSurface(
                optional_property_quantity(
                    declaration,
                    statements,
                    "reference_offset",
                    PhysicalDimension::LENGTH,
                    diagnostics,
                )?
                .or_else(|| {
                    diagnostic(
                        declaration,
                        declaration.span,
                        "offset reference surface requires `reference_offset`",
                        diagnostics,
                    );
                    None
                })?,
            ),
            value => {
                diagnostic(
                    declaration,
                    declaration.span,
                    format!("unsupported shell reference surface `{value}`"),
                    diagnostics,
                );
                return None;
            }
        };
    if !matches!(
        reference_surface,
        ShellReferenceSurface::OffsetFromMidSurface(_)
    ) && optional_property_assignment(statements, "reference_offset").is_some()
    {
        diagnostic(
            declaration,
            declaration.span,
            "`reference_offset` is valid only with `offset_from_mid_surface`",
            diagnostics,
        );
        return None;
    }
    let shear_correction =
        optional_property_scalar(declaration, statements, "shear_correction", diagnostics)?;
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
            &["material", "thickness", "angle"],
            diagnostics,
        );
        let material = required_assignment(declaration, body, "material", diagnostics)?;
        plies.push(AuthoredPly {
            ply_id: ply_id.trim().into(),
            material: material_source(bound, declaration, material, diagnostics)?,
            thickness_m: runmat_types::physical::Length::new(required_length(
                declaration,
                body,
                "thickness",
                diagnostics,
            )?),
            angle_rad:
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Angle, _>::new(
                    required_quantity(
                        declaration,
                        body,
                        "angle",
                        // A ply angle is an Angle quantity (`45 deg`, `0 rad`), not a bare
                        // dimensionless number.
                        runmat_types::physical::DynamicDimension::quantity_type(
                            runmat_types::physical::QuantityKind::Angle,
                        ),
                        diagnostics,
                    )?,
                ),
        });
    }
    Some(AuthoredLayupDefinition {
        layup_id: id.into(),
        reference_orientation,
        ply_order,
        reference_surface,
        shear_correction,
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
                if *relation != runmat_design_syntax::FeaAssignmentRelation::Equal
                    || !matches!(
                        name.as_str(),
                        "reference_orientation"
                            | "ply_order"
                            | "reference_surface"
                            | "reference_offset"
                            | "shear_correction"
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
                "layup accepts only its declared settings and ordered ply blocks",
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

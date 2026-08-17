use super::definition_validation_math::*;
use super::nurbs_validation::{
    validate_nurbs_curve2, validate_nurbs_curve3, validate_nurbs_surface,
};
use super::{
    BodyMassProperties, ExactCurveDefinition, ExactCurveImplementation,
    ExactMassPropertiesImplementation, ExactPcurveDefinition, ExactPcurveImplementation,
    ExactSurfaceDefinition, ExactSurfaceImplementation, ExactTrimClassifierImplementation,
    KernelEvaluatorRef, ParameterRange,
};
use crate::model::GeometryContractError;

pub(super) fn validate_curve(
    implementation: &ExactCurveImplementation,
) -> Result<(), GeometryContractError> {
    match implementation {
        ExactCurveImplementation::Portable { definition } => match definition {
            ExactCurveDefinition::Line {
                origin_m,
                direction_m_per_parameter,
                domain,
            } => {
                finite_vector("line origin", origin_m)?;
                nonzero_vector("line direction", direction_m_per_parameter)?;
                validate_range("line domain", domain)
            }
            ExactCurveDefinition::Circle {
                center_m,
                x_axis,
                y_axis,
                radius_m,
                domain,
            } => {
                finite_vector("circle center", center_m)?;
                orthonormal_pair("circle axes", x_axis, y_axis)?;
                positive("circle radius", *radius_m)?;
                validate_range("circle domain", domain)
            }
            ExactCurveDefinition::Ellipse {
                center_m,
                x_axis,
                y_axis,
                major_radius_m,
                minor_radius_m,
                domain,
            } => {
                finite_vector("ellipse center", center_m)?;
                orthonormal_pair("ellipse axes", x_axis, y_axis)?;
                positive("ellipse major radius", *major_radius_m)?;
                positive("ellipse minor radius", *minor_radius_m)?;
                if major_radius_m < minor_radius_m {
                    return Err(invalid(
                        "ellipse radii",
                        "major radius must not be smaller than minor radius",
                    ));
                }
                validate_range("ellipse domain", domain)
            }
            ExactCurveDefinition::Nurbs { definition } => validate_nurbs_curve3(definition),
        },
        ExactCurveImplementation::Kernel { reference } => validate_kernel_ref(reference),
    }
}

pub(super) fn validate_pcurve(
    implementation: &ExactPcurveImplementation,
) -> Result<(), GeometryContractError> {
    match implementation {
        ExactPcurveImplementation::Portable { definition } => match definition {
            ExactPcurveDefinition::Line {
                origin_uv,
                direction_uv_per_parameter,
                domain,
            } => {
                finite_vector("pcurve line origin", origin_uv)?;
                nonzero_vector("pcurve line direction", direction_uv_per_parameter)?;
                validate_range("pcurve line domain", domain)
            }
            ExactPcurveDefinition::Circle {
                center_uv,
                x_axis_uv,
                y_axis_uv,
                radius_uv,
                domain,
            } => {
                finite_vector("pcurve circle center", center_uv)?;
                orthonormal_pair("pcurve circle axes", x_axis_uv, y_axis_uv)?;
                positive("pcurve circle radius", *radius_uv)?;
                validate_range("pcurve circle domain", domain)
            }
            ExactPcurveDefinition::Nurbs { definition } => validate_nurbs_curve2(definition),
        },
        ExactPcurveImplementation::Kernel { reference } => validate_kernel_ref(reference),
    }
}

pub(super) fn validate_surface(
    implementation: &ExactSurfaceImplementation,
) -> Result<(), GeometryContractError> {
    match implementation {
        ExactSurfaceImplementation::Portable { definition } => match definition {
            ExactSurfaceDefinition::Plane {
                origin_m,
                u_axis_m_per_parameter,
                v_axis_m_per_parameter,
                domains,
            } => {
                finite_vector("plane origin", origin_m)?;
                independent_pair(
                    "plane parameter axes",
                    u_axis_m_per_parameter,
                    v_axis_m_per_parameter,
                )?;
                validate_ranges("plane domains", domains)
            }
            ExactSurfaceDefinition::Cylinder {
                origin_m,
                x_axis,
                y_axis,
                axis_m_per_v,
                radius_m,
                domains,
            } => {
                finite_vector("cylinder origin", origin_m)?;
                orthonormal_triple("cylinder frame", x_axis, y_axis, axis_m_per_v)?;
                positive("cylinder radius", *radius_m)?;
                validate_ranges("cylinder domains", domains)
            }
            ExactSurfaceDefinition::Cone {
                apex_m,
                x_axis,
                y_axis,
                axis,
                semi_angle_rad,
                domains,
            } => {
                finite_vector("cone apex", apex_m)?;
                orthonormal_triple("cone frame", x_axis, y_axis, axis)?;
                if !semi_angle_rad.is_finite()
                    || *semi_angle_rad <= 0.0
                    || *semi_angle_rad >= std::f64::consts::FRAC_PI_2
                {
                    return Err(invalid(
                        "cone semi-angle",
                        "must be finite and strictly between zero and pi/2",
                    ));
                }
                validate_ranges("cone domains", domains)
            }
            ExactSurfaceDefinition::Sphere {
                center_m,
                x_axis,
                y_axis,
                z_axis,
                radius_m,
                domains,
            } => {
                finite_vector("sphere center", center_m)?;
                orthonormal_triple("sphere frame", x_axis, y_axis, z_axis)?;
                positive("sphere radius", *radius_m)?;
                validate_ranges("sphere domains", domains)
            }
            ExactSurfaceDefinition::Torus {
                center_m,
                x_axis,
                y_axis,
                z_axis,
                major_radius_m,
                minor_radius_m,
                domains,
            } => {
                finite_vector("torus center", center_m)?;
                orthonormal_triple("torus frame", x_axis, y_axis, z_axis)?;
                positive("torus major radius", *major_radius_m)?;
                positive("torus minor radius", *minor_radius_m)?;
                if major_radius_m <= minor_radius_m {
                    return Err(invalid(
                        "torus radii",
                        "ring torus major radius must exceed minor radius",
                    ));
                }
                validate_ranges("torus domains", domains)
            }
            ExactSurfaceDefinition::Nurbs { definition } => validate_nurbs_surface(definition),
        },
        ExactSurfaceImplementation::Kernel { reference } => validate_kernel_ref(reference),
    }
}

pub(super) fn validate_trim_classifier(
    implementation: &ExactTrimClassifierImplementation,
) -> Result<(), GeometryContractError> {
    match implementation {
        ExactTrimClassifierImplementation::OrientedPcurveWinding => Ok(()),
        ExactTrimClassifierImplementation::Kernel { reference } => validate_kernel_ref(reference),
    }
}

pub(super) fn validate_mass_properties(
    implementation: &ExactMassPropertiesImplementation,
) -> Result<(), GeometryContractError> {
    match implementation {
        ExactMassPropertiesImplementation::Kernel { reference } => validate_kernel_ref(reference),
        ExactMassPropertiesImplementation::KernelValidated {
            properties,
            validation_digest,
        } => {
            validate_mass_values(properties)?;
            if *validation_digest == [0; 32] {
                return Err(invalid(
                    "mass-properties validation digest",
                    "must bind independent kernel validation evidence",
                ));
            }
            Ok(())
        }
    }
}

pub(super) fn curve_is_periodic(implementation: &ExactCurveImplementation) -> Option<bool> {
    match implementation {
        ExactCurveImplementation::Portable { definition } => Some(match definition {
            ExactCurveDefinition::Circle { .. } | ExactCurveDefinition::Ellipse { .. } => true,
            ExactCurveDefinition::Line { .. } => false,
            ExactCurveDefinition::Nurbs { definition } => definition.periodic,
        }),
        ExactCurveImplementation::Kernel { .. } => None,
    }
}

pub(super) fn surface_periodicity(
    implementation: &ExactSurfaceImplementation,
) -> Option<[bool; 2]> {
    match implementation {
        ExactSurfaceImplementation::Portable { definition } => Some(match definition {
            ExactSurfaceDefinition::Plane { .. } => [false, false],
            ExactSurfaceDefinition::Cylinder { .. }
            | ExactSurfaceDefinition::Cone { .. }
            | ExactSurfaceDefinition::Sphere { .. } => [true, false],
            ExactSurfaceDefinition::Torus { .. } => [true, true],
            ExactSurfaceDefinition::Nurbs { definition } => {
                [definition.periodic_u, definition.periodic_v]
            }
        }),
        ExactSurfaceImplementation::Kernel { .. } => None,
    }
}

pub(super) fn curve_domain(implementation: &ExactCurveImplementation) -> Option<ParameterRange> {
    match implementation {
        ExactCurveImplementation::Portable { definition } => Some(match definition {
            ExactCurveDefinition::Line { domain, .. }
            | ExactCurveDefinition::Circle { domain, .. }
            | ExactCurveDefinition::Ellipse { domain, .. } => *domain,
            ExactCurveDefinition::Nurbs { definition } => definition.domain,
        }),
        ExactCurveImplementation::Kernel { .. } => None,
    }
}

pub(super) fn pcurve_domain(implementation: &ExactPcurveImplementation) -> Option<ParameterRange> {
    match implementation {
        ExactPcurveImplementation::Portable { definition } => Some(match definition {
            ExactPcurveDefinition::Line { domain, .. }
            | ExactPcurveDefinition::Circle { domain, .. } => *domain,
            ExactPcurveDefinition::Nurbs { definition } => definition.domain,
        }),
        ExactPcurveImplementation::Kernel { .. } => None,
    }
}

pub(super) fn curve_dynamic_value_count(implementation: &ExactCurveImplementation) -> usize {
    match implementation {
        ExactCurveImplementation::Portable {
            definition: ExactCurveDefinition::Nurbs { definition },
        } => definition
            .knots
            .len()
            .saturating_add(definition.weights.len())
            .saturating_add(definition.control_points_m.len().saturating_mul(3)),
        _ => 0,
    }
}

pub(super) fn pcurve_dynamic_value_count(implementation: &ExactPcurveImplementation) -> usize {
    match implementation {
        ExactPcurveImplementation::Portable {
            definition: ExactPcurveDefinition::Nurbs { definition },
        } => definition
            .knots
            .len()
            .saturating_add(definition.weights.len())
            .saturating_add(definition.control_points_uv.len().saturating_mul(2)),
        _ => 0,
    }
}

pub(super) fn surface_dynamic_value_count(implementation: &ExactSurfaceImplementation) -> usize {
    match implementation {
        ExactSurfaceImplementation::Portable {
            definition: ExactSurfaceDefinition::Nurbs { definition },
        } => definition
            .u_knots
            .len()
            .saturating_add(definition.v_knots.len())
            .saturating_add(definition.weights.len())
            .saturating_add(definition.control_points_m.len().saturating_mul(3)),
        _ => 0,
    }
}

fn validate_mass_values(properties: &BodyMassProperties) -> Result<(), GeometryContractError> {
    if !properties.volume_m3.is_finite()
        || properties.volume_m3 < 0.0
        || !properties.surface_area_m2.is_finite()
        || properties.surface_area_m2 <= 0.0
    {
        return Err(invalid(
            "body mass properties",
            "volume must be finite and non-negative; surface area must be finite and positive",
        ));
    }
    finite_vector("body centroid", &properties.centroid_m)?;
    if properties
        .inertia_about_centroid_m5
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(invalid("body inertia", "all entries must be finite"));
    }
    let [ixx, iyy, izz, ..] = properties.inertia_about_centroid_m5;
    if ixx < 0.0 || iyy < 0.0 || izz < 0.0 || ixx + iyy < izz || ixx + izz < iyy || iyy + izz < ixx
    {
        return Err(invalid(
            "body inertia",
            "principal diagonal moments must be non-negative and satisfy triangle inequalities",
        ));
    }
    Ok(())
}

fn validate_kernel_ref(reference: &KernelEvaluatorRef) -> Result<(), GeometryContractError> {
    validate_token("kernel evaluator entity token", &reference.entity_token)?;
    if reference.representation_digest == [0; 32] {
        return Err(invalid(
            "kernel evaluator representation digest",
            "must be nonzero",
        ));
    }
    Ok(())
}

pub(super) fn validate_token(field: &str, value: &str) -> Result<(), GeometryContractError> {
    if value.is_empty()
        || value.len() > 512
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.trim() != value
    {
        return Err(invalid(field, "must be a bounded printable ASCII token"));
    }
    Ok(())
}

fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}

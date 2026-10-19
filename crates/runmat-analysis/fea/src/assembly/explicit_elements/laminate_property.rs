use runmat_analysis_core::{
    LaminateShellSectionModel, MaterialElasticityModel, MechanicalDampingModel, PlyOrderDirection,
    ShellReferenceSurface,
};

use crate::{
    assembly::elements::shell::{
        laminate::{
            LaminaElasticity, LaminateDefinition, LaminateLayer, LaminateMassProperties,
            LaminatePlyOrder, LaminateReferenceSurface,
        },
        ShellMassMoments,
    },
    properties::CompiledLaminatePly,
};

pub(super) fn definition(
    section: &LaminateShellSectionModel,
    plies: &[CompiledLaminatePly],
) -> Result<LaminateDefinition, String> {
    if plies.len() != section.plies.len() {
        return Err(format!(
            "laminate section `{}` compiled {} plies for {} definitions",
            section.section_id,
            plies.len(),
            section.plies.len()
        ));
    }
    let layers = plies
        .iter()
        .map(layer)
        .collect::<Result<Vec<_>, String>>()?;
    Ok(LaminateDefinition {
        layers,
        ply_order: match section.ply_order {
            PlyOrderDirection::AlongSurfaceNormal => LaminatePlyOrder::AlongSurfaceNormal,
            PlyOrderDirection::OppositeSurfaceNormal => LaminatePlyOrder::OppositeSurfaceNormal,
        },
        reference_surface: match section.reference_surface {
            ShellReferenceSurface::MidSurface => LaminateReferenceSurface::Mid,
            ShellReferenceSurface::FirstPlySurface => LaminateReferenceSurface::FirstPly,
            ShellReferenceSurface::LastPlySurface => LaminateReferenceSurface::LastPly,
            ShellReferenceSurface::OffsetFromMidSurface(offset) => {
                LaminateReferenceSurface::OffsetFromMid(offset)
            }
        },
        shear_correction: section.shear_correction,
    })
}

fn layer(ply: &CompiledLaminatePly) -> Result<LaminateLayer, String> {
    let elasticity = match ply
        .material
        .require_lamina_elasticity()
        .map_err(|error| error.to_string())?
    {
        MaterialElasticityModel::IsotropicLinear(value) => {
            let shear = value.youngs_modulus_pa / (2.0 * (1.0 + value.poisson_ratio));
            LaminaElasticity {
                longitudinal_modulus_pa: value.youngs_modulus_pa,
                transverse_modulus_pa: value.youngs_modulus_pa,
                major_poisson_ratio: value.poisson_ratio,
                in_plane_shear_modulus_pa: shear,
                transverse_shear_modulus_13_pa: shear,
                transverse_shear_modulus_23_pa: shear,
            }
        }
        MaterialElasticityModel::OrthotropicLamina(value) => LaminaElasticity {
            longitudinal_modulus_pa: value.longitudinal_modulus_pa,
            transverse_modulus_pa: value.transverse_modulus_pa,
            major_poisson_ratio: value.major_poisson_ratio,
            in_plane_shear_modulus_pa: value.in_plane_shear_modulus_pa,
            transverse_shear_modulus_13_pa: value.transverse_shear_modulus_13_pa,
            transverse_shear_modulus_23_pa: value.transverse_shear_modulus_23_pa,
        },
    };
    let density_kg_per_m3 = ply
        .material
        .require_mass_density()
        .map_err(|error| error.to_string())?;
    let (mass_damping_per_second, stiffness_damping_seconds) =
        match ply.material.damping_or_undamped() {
            MechanicalDampingModel::Undamped => (0.0, 0.0),
            MechanicalDampingModel::Rayleigh {
                mass_coefficient,
                stiffness_coefficient,
            } => (mass_coefficient.per_second, stiffness_coefficient.seconds),
        };
    Ok(LaminateLayer {
        layer_id: ply.definition.ply_id.clone(),
        thickness_m: ply.definition.thickness_m,
        angle_rad: ply.definition.angle_rad,
        elasticity,
        density_kg_per_m3,
        mass_damping_per_second,
        stiffness_damping_seconds,
    })
}

pub(super) fn mass_moments(mass: LaminateMassProperties) -> (ShellMassMoments, ShellMassMoments) {
    (mass.inertia, mass.mass_damping)
}

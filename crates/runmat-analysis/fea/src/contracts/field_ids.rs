use serde::{Deserialize, Serialize};

pub const FEA_FIELD_STRUCTURAL_DISPLACEMENT: &str = "structural.displacement";
pub const FEA_FIELD_STRUCTURAL_ROTATION: &str = "structural.rotation";
pub const FEA_FIELD_STRUCTURAL_VON_MISES: &str = "structural.von_mises";
pub const FEA_FIELD_STRUCTURAL_NODAL_VON_MISES: &str = "structural.nodal_von_mises";
pub const FEA_FIELD_STRUCTURAL_NODAL_STRESS: &str = "structural.nodal_stress";
pub const FEA_FIELD_STRUCTURAL_STRAIN: &str = "structural.strain";
pub const FEA_FIELD_STRUCTURAL_STRESS: &str = "structural.stress";
pub const FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN: &str =
    "structural.integration_point_strain";
pub const FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS: &str =
    "structural.integration_point_stress";
pub const FEA_FIELD_STRUCTURAL_REACTION_FORCE: &str = "structural.reaction_force";
pub const FEA_FIELD_STRUCTURAL_REACTION_MOMENT: &str = "structural.reaction_moment";
pub const FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE: &str =
    "structural.constraint_reaction_force";
pub const FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN: &str =
    "structural.constraint_reaction_moment_about_global_origin";
pub const FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE: &str = "structural.connector_force";
pub const FEA_FIELD_STRUCTURAL_CONNECTOR_STORED_ENERGY: &str = "structural.connector_stored_energy";
pub const FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY: &str = "structural.strain_energy_density";
pub const FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY: &str = "structural.total_strain_energy";
pub const FEA_FIELD_STRUCTURAL_RESIDUAL_NORM: &str = "structural.residual_norm";
pub const FEA_FIELD_STRUCTURAL_EQUATION_SCALE: &str = "structural.equation_scale";
pub const FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE: &str = "structural.beam_axial_force";
pub const FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE: &str = "structural.beam_shear_force";
pub const FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT: &str = "structural.beam_torsion_moment";
pub const FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT: &str = "structural.beam_bending_moment";
pub const FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS: &str = "structural.beam_bending_stress";
pub const FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS: &str = "structural.beam_torsion_stress";
pub const FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE: &str = "structural.shell_membrane_force";
pub const FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT: &str = "structural.shell_bending_moment";
pub const FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR: &str = "structural.shell_transverse_shear";
pub const FEA_FIELD_STRUCTURAL_SHELL_VON_MISES: &str = "structural.shell_von_mises";

const FEA_FIELD_RESPONSE_SPECTRUM_PREFIX: &str = "structural.response_spectrum";
const FEA_FIELD_RANDOM_RESPONSE_PREFIX: &str = "structural.random_response";

/// Closed structural field selector used by language and storage boundaries.
///
/// Field identifiers remain stable data names, while callers select supported structural
/// quantities through this enum rather than dispatching solver behavior from strings.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralFieldSelector {
    Displacement,
    Rotation,
    VonMises,
    NodalVonMises,
    NodalStress,
    Strain,
    Stress,
    IntegrationPointStrain,
    IntegrationPointStress,
    ReactionForce,
    ReactionMoment,
    ConstraintReactionForce,
    ConstraintReactionMomentAboutGlobalOrigin,
    StrainEnergyDensity,
    TotalStrainEnergy,
    ResidualNorm,
    EquationScale,
    BeamAxialForce,
    BeamShearForce,
    BeamTorsionMoment,
    BeamBendingMoment,
    BeamBendingStress,
    BeamTorsionStress,
    ShellMembraneForce,
    ShellBendingMoment,
    ShellTransverseShear,
    ShellVonMises,
}

impl StructuralFieldSelector {
    pub const fn field_id(self) -> &'static str {
        match self {
            Self::Displacement => FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            Self::Rotation => FEA_FIELD_STRUCTURAL_ROTATION,
            Self::VonMises => FEA_FIELD_STRUCTURAL_VON_MISES,
            Self::NodalVonMises => FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
            Self::NodalStress => FEA_FIELD_STRUCTURAL_NODAL_STRESS,
            Self::Strain => FEA_FIELD_STRUCTURAL_STRAIN,
            Self::Stress => FEA_FIELD_STRUCTURAL_STRESS,
            Self::IntegrationPointStrain => FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN,
            Self::IntegrationPointStress => FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
            Self::ReactionForce => FEA_FIELD_STRUCTURAL_REACTION_FORCE,
            Self::ReactionMoment => FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
            Self::ConstraintReactionForce => FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE,
            Self::ConstraintReactionMomentAboutGlobalOrigin => {
                FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN
            }
            Self::StrainEnergyDensity => FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
            Self::TotalStrainEnergy => FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
            Self::ResidualNorm => FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
            Self::EquationScale => FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
            Self::BeamAxialForce => FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE,
            Self::BeamShearForce => FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
            Self::BeamTorsionMoment => FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT,
            Self::BeamBendingMoment => FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT,
            Self::BeamBendingStress => FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS,
            Self::BeamTorsionStress => FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS,
            Self::ShellMembraneForce => FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
            Self::ShellBendingMoment => FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
            Self::ShellTransverseShear => FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
            Self::ShellVonMises => FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
        }
    }
}

/// Closed selector for a response-spectrum statistical field.
///
/// Nodal quantities preserve the configured spectrum ordinate. Deformation quantities are
/// component-magnitude envelopes recovered from a displacement-convertible response and must not
/// be confused with a simultaneous signed structural state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseSpectrumFieldSelector {
    Nodal(crate::solve::dynamics::ResponseSpectrumNodalQuantity),
    Deformation(crate::post::fields::StructuralLinearFieldQuantity),
}

impl ResponseSpectrumFieldSelector {
    pub fn field_id(self) -> String {
        match self {
            Self::Nodal(quantity) => format!(
                "{FEA_FIELD_RESPONSE_SPECTRUM_PREFIX}.nodal.{}",
                response_spectrum_nodal_quantity_name(quantity)
            ),
            Self::Deformation(quantity) => format!(
                "{FEA_FIELD_RESPONSE_SPECTRUM_PREFIX}.envelope.{}",
                structural_linear_quantity_name(quantity)
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomResponseStatistic {
    RootMeanSquare,
    Variance,
}

/// Closed selector for a random-response statistical field.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RandomResponseFieldSelector {
    statistic: RandomResponseStatistic,
    quantity: crate::post::fields::StructuralLinearFieldQuantity,
}

impl RandomResponseFieldSelector {
    pub const fn root_mean_square(
        quantity: crate::post::fields::StructuralLinearFieldQuantity,
    ) -> Self {
        Self {
            statistic: RandomResponseStatistic::RootMeanSquare,
            quantity,
        }
    }

    pub const fn variance(quantity: crate::post::fields::StructuralLinearFieldQuantity) -> Self {
        Self {
            statistic: RandomResponseStatistic::Variance,
            quantity,
        }
    }

    pub const fn statistic(self) -> RandomResponseStatistic {
        self.statistic
    }

    pub const fn quantity(self) -> crate::post::fields::StructuralLinearFieldQuantity {
        self.quantity
    }

    pub fn field_id(self) -> String {
        let statistic = match self.statistic {
            RandomResponseStatistic::RootMeanSquare => "rms",
            RandomResponseStatistic::Variance => "variance",
        };
        format!(
            "{FEA_FIELD_RANDOM_RESPONSE_PREFIX}.{statistic}.{}",
            structural_linear_quantity_name(self.quantity)
        )
    }
}

pub const fn structural_linear_quantity_for_selector(
    selector: StructuralFieldSelector,
) -> Option<crate::post::fields::StructuralLinearFieldQuantity> {
    use crate::post::fields::StructuralLinearFieldQuantity as Quantity;
    match selector {
        StructuralFieldSelector::Displacement => Some(Quantity::Displacement),
        StructuralFieldSelector::Rotation => Some(Quantity::Rotation),
        StructuralFieldSelector::Strain => Some(Quantity::Strain),
        StructuralFieldSelector::Stress => Some(Quantity::Stress),
        StructuralFieldSelector::ShellMembraneForce => Some(Quantity::ShellMembraneForce),
        StructuralFieldSelector::ShellBendingMoment => Some(Quantity::ShellBendingMoment),
        StructuralFieldSelector::ShellTransverseShear => Some(Quantity::ShellTransverseShear),
        _ => None,
    }
}

pub const fn structural_linear_quantity_name(
    quantity: crate::post::fields::StructuralLinearFieldQuantity,
) -> &'static str {
    use crate::post::fields::StructuralLinearFieldQuantity as Quantity;
    match quantity {
        Quantity::Displacement => "displacement",
        Quantity::Rotation => "rotation",
        Quantity::Strain => "strain",
        Quantity::Stress => "stress",
        Quantity::ShellMembraneForce => "shell_membrane_force",
        Quantity::ShellBendingMoment => "shell_bending_moment",
        Quantity::ShellTransverseShear => "shell_transverse_shear",
    }
}

pub const fn response_spectrum_nodal_quantity_name(
    quantity: crate::solve::dynamics::ResponseSpectrumNodalQuantity,
) -> &'static str {
    use crate::solve::dynamics::ResponseSpectrumNodalQuantity as Quantity;
    match quantity {
        Quantity::RelativeDisplacement => "relative_displacement",
        Quantity::Rotation => "rotation",
        Quantity::RelativeVelocity => "relative_velocity",
        Quantity::RelativeAngularVelocity => "relative_angular_velocity",
        Quantity::RelativeAcceleration => "relative_acceleration",
        Quantity::RelativeAngularAcceleration => "relative_angular_acceleration",
        Quantity::AbsoluteAcceleration => "absolute_acceleration",
        Quantity::AbsoluteAngularAcceleration => "absolute_angular_acceleration",
        Quantity::PseudoVelocity => "pseudo_velocity",
        Quantity::PseudoAngularVelocity => "pseudo_angular_velocity",
        Quantity::PseudoAcceleration => "pseudo_acceleration",
        Quantity::PseudoAngularAcceleration => "pseudo_angular_acceleration",
    }
}

pub fn fea_modal_mode_shape_field_id(mode_number: usize) -> String {
    format!("modal.mode_shape.{mode_number}")
}

pub fn fea_buckling_mode_shape_field_id(mode_number: usize) -> String {
    format!("buckling.mode_shape.{mode_number}")
}

pub fn fea_buckling_rotation_shape_field_id(mode_number: usize) -> String {
    format!("buckling.rotation_shape.{mode_number}")
}

pub const FEA_FIELD_BUCKLING_LOAD_FACTOR: &str = "buckling.load_factor";
pub const FEA_FIELD_BUCKLING_RESIDUAL_NORM: &str = "buckling.residual_norm";

pub const FEA_FIELD_MODAL_FREQUENCY_HZ: &str = "modal.frequency_hz";
pub const FEA_FIELD_MODAL_EIGENVALUE: &str = "modal.eigenvalue";
pub const FEA_FIELD_MODAL_MODAL_MASS: &str = "modal.modal_mass";
pub const FEA_FIELD_MODAL_MODAL_STIFFNESS: &str = "modal.modal_stiffness";
pub const FEA_FIELD_MODAL_PARTICIPATION_FACTOR: &str = "modal.participation_factor";
pub const FEA_FIELD_MODAL_EFFECTIVE_MASS: &str = "modal.effective_mass";
pub const FEA_FIELD_MODAL_CUMULATIVE_EFFECTIVE_MASS: &str = "modal.cumulative_effective_mass";
pub const FEA_FIELD_MODAL_CUMULATIVE_MASS_RATIO: &str = "modal.cumulative_mass_ratio";
pub const FEA_FIELD_MODAL_TOTAL_FREE_MASS: &str = "modal.total_free_mass";
pub const FEA_FIELD_MODAL_RESIDUAL_NORM: &str = "modal.residual_norm";
pub const FEA_FIELD_MODAL_BACKWARD_ERROR: &str = "modal.backward_error";
pub const FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION: &str =
    "modal.relative_frequency_separation";
pub const FEA_FIELD_MODAL_M_ORTHOGONALITY: &str = "modal.m_orthogonality";

pub const FEA_FIELD_ACOUSTIC_PRESSURE_REAL: &str = "acoustic.pressure_real";
pub const FEA_FIELD_ACOUSTIC_PRESSURE_IMAG: &str = "acoustic.pressure_imag";
pub const FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE: &str = "acoustic.pressure_magnitude";
pub const FEA_FIELD_ACOUSTIC_PHASE: &str = "acoustic.phase";
pub const FEA_FIELD_ACOUSTIC_SOUND_PRESSURE_LEVEL_DB: &str = "acoustic.sound_pressure_level_db";
pub const FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY: &str = "acoustic.particle_velocity";

pub fn fea_acoustic_frequency_response_field_id(frequency_hz: f64) -> String {
    let scaled_hz = (frequency_hz.max(0.0) * 1_000.0).round() as u64;
    format!("acoustic.frequency_response.hz_x1000_{scaled_hz}")
}

pub const FEA_FIELD_CFD_VELOCITY: &str = "cfd.velocity";
pub const FEA_FIELD_CFD_PRESSURE: &str = "cfd.pressure";
pub const FEA_FIELD_CFD_VORTICITY: &str = "cfd.vorticity";
pub const FEA_FIELD_CFD_WALL_SHEAR_STRESS: &str = "cfd.wall_shear_stress";
pub const FEA_FIELD_CFD_FACE_PRESSURE: &str = "cfd.face_pressure";
pub const FEA_FIELD_CFD_RESIDUAL_MOMENTUM: &str = "cfd.residual_momentum";
pub const FEA_FIELD_CFD_RESIDUAL_CONTINUITY: &str = "cfd.residual_continuity";
pub const FEA_FIELD_CFD_REYNOLDS_NUMBER: &str = "cfd.reynolds_number";

pub const FEA_FIELD_CHT_FLUID_VELOCITY: &str = "cht.fluid_velocity";
pub const FEA_FIELD_CHT_FLUID_PRESSURE: &str = "cht.fluid_pressure";

pub fn fea_cht_fluid_temperature_field_id(step_index: usize) -> String {
    format!("cht.fluid_temperature.{step_index}")
}

pub fn fea_cht_solid_temperature_field_id(step_index: usize) -> String {
    format!("cht.solid_temperature.{step_index}")
}

pub fn fea_cht_interface_heat_flux_field_id(step_index: usize) -> String {
    format!("cht.interface_heat_flux.{step_index}")
}

pub fn fea_cht_interface_temperature_jump_field_id(step_index: usize) -> String {
    format!("cht.interface_temperature_jump.{step_index}")
}

pub fn fea_cht_energy_residual_field_id(step_index: usize) -> String {
    format!("cht.energy_residual.{step_index}")
}

pub fn fea_fsi_fluid_velocity_field_id(step_index: usize) -> String {
    format!("fsi.fluid_velocity.{step_index}")
}

pub fn fea_fsi_fluid_pressure_field_id(step_index: usize) -> String {
    format!("fsi.fluid_pressure.{step_index}")
}

pub fn fea_fsi_structural_displacement_field_id(step_index: usize) -> String {
    format!("fsi.structural_displacement.{step_index}")
}

pub fn fea_fsi_interface_pressure_field_id(step_index: usize) -> String {
    format!("fsi.interface_pressure.{step_index}")
}

pub fn fea_fsi_interface_traction_field_id(step_index: usize) -> String {
    format!("fsi.interface_traction.{step_index}")
}

pub fn fea_fsi_interface_displacement_field_id(step_index: usize) -> String {
    format!("fsi.interface_displacement.{step_index}")
}

pub fn fea_fsi_interface_residual_field_id(step_index: usize) -> String {
    format!("fsi.interface_residual.{step_index}")
}

pub fn fea_fsi_coupling_iteration_count_field_id(step_index: usize) -> String {
    format!("fsi.coupling_iteration_count.{step_index}")
}

pub fn fea_thermal_temperature_field_id(snapshot_index: usize) -> String {
    format!("thermal.temperature.{snapshot_index}")
}

/// Second-side temperature at the nodes of finite-conductance thermal interfaces.
pub fn fea_thermal_interface_temperature_field_id(snapshot_index: usize) -> String {
    format!("thermal.interface_second_side_temperature.{snapshot_index}")
}

pub fn fea_thermal_temperature_gradient_field_id(snapshot_index: usize) -> String {
    format!("thermal.temperature_gradient.{snapshot_index}")
}

pub fn fea_thermal_heat_flux_field_id(snapshot_index: usize) -> String {
    format!("thermal.heat_flux.{snapshot_index}")
}

pub fn fea_thermal_energy_storage_rate_field_id(snapshot_index: usize) -> String {
    format!("thermal.energy_storage_rate.{snapshot_index}")
}

/// Structural fields that have one snapshot field per retained transient state.
pub const FEA_TRANSIENT_SNAPSHOT_STRUCTURAL_FIELDS: [&str; 8] = [
    FEA_FIELD_STRUCTURAL_DISPLACEMENT,
    FEA_FIELD_STRUCTURAL_ROTATION,
    FEA_FIELD_STRUCTURAL_STRAIN,
    FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_VON_MISES,
    FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
    FEA_FIELD_STRUCTURAL_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
];

/// The accepted-state snapshot field that carries a structural field at one retained transient
/// time. Transient recovery names its snapshots through this map, and time-sampled field reads
/// resolve through it, so the two cannot disagree.
pub fn fea_transient_snapshot_field_id(field_id: &str, snapshot_index: usize) -> Option<String> {
    let family = match field_id {
        FEA_FIELD_STRUCTURAL_DISPLACEMENT => fea_transient_displacement_field_id,
        FEA_FIELD_STRUCTURAL_ROTATION => fea_transient_rotation_field_id,
        FEA_FIELD_STRUCTURAL_STRAIN => fea_transient_strain_field_id,
        FEA_FIELD_STRUCTURAL_STRESS => fea_transient_stress_field_id,
        FEA_FIELD_STRUCTURAL_VON_MISES => fea_transient_von_mises_field_id,
        FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY => fea_transient_strain_energy_density_field_id,
        FEA_FIELD_STRUCTURAL_REACTION_FORCE => fea_transient_reaction_force_field_id,
        FEA_FIELD_STRUCTURAL_REACTION_MOMENT => fea_transient_reaction_moment_field_id,
        _ => return None,
    };
    Some(family(snapshot_index))
}

pub fn fea_transient_displacement_field_id(snapshot_index: usize) -> String {
    format!("transient.displacement.{snapshot_index}")
}

pub fn fea_transient_rotation_field_id(snapshot_index: usize) -> String {
    format!("transient.rotation.{snapshot_index}")
}

pub fn fea_transient_velocity_field_id(snapshot_index: usize) -> String {
    format!("transient.velocity.{snapshot_index}")
}

pub fn fea_transient_angular_velocity_field_id(snapshot_index: usize) -> String {
    format!("transient.angular_velocity.{snapshot_index}")
}

pub fn fea_transient_acceleration_field_id(snapshot_index: usize) -> String {
    format!("transient.acceleration.{snapshot_index}")
}

pub fn fea_transient_angular_acceleration_field_id(snapshot_index: usize) -> String {
    format!("transient.angular_acceleration.{snapshot_index}")
}

pub fn fea_transient_von_mises_field_id(snapshot_index: usize) -> String {
    format!("transient.von_mises.{snapshot_index}")
}

pub fn fea_transient_strain_field_id(snapshot_index: usize) -> String {
    format!("transient.strain.{snapshot_index}")
}

pub fn fea_transient_stress_field_id(snapshot_index: usize) -> String {
    format!("transient.stress.{snapshot_index}")
}

pub fn fea_transient_strain_energy_density_field_id(snapshot_index: usize) -> String {
    format!("transient.strain_energy_density.{snapshot_index}")
}

pub fn fea_transient_kinetic_energy_field_id(snapshot_index: usize) -> String {
    format!("transient.kinetic_energy.{snapshot_index}")
}

pub fn fea_transient_strain_energy_field_id(snapshot_index: usize) -> String {
    format!("transient.strain_energy.{snapshot_index}")
}

pub fn fea_transient_residual_norm_field_id(snapshot_index: usize) -> String {
    format!("transient.residual_norm.{snapshot_index}")
}

pub fn fea_transient_reaction_force_field_id(snapshot_index: usize) -> String {
    format!("transient.reaction_force.{snapshot_index}")
}

pub fn fea_transient_reaction_moment_field_id(snapshot_index: usize) -> String {
    format!("transient.reaction_moment.{snapshot_index}")
}

pub fn fea_transient_connector_force_field_id(snapshot_index: usize) -> String {
    format!("transient.connector_force.{snapshot_index}")
}

pub fn fea_transient_connector_stored_energy_field_id(snapshot_index: usize) -> String {
    format!("transient.connector_stored_energy.{snapshot_index}")
}

pub fn fea_transient_external_work_field_id(snapshot_index: usize) -> String {
    format!("transient.external_work.{snapshot_index}")
}

pub fn fea_transient_support_work_field_id(snapshot_index: usize) -> String {
    format!("transient.support_work.{snapshot_index}")
}

pub fn fea_transient_damping_dissipation_field_id(snapshot_index: usize) -> String {
    format!("transient.damping_dissipation.{snapshot_index}")
}

pub fn fea_transient_energy_balance_residual_field_id(snapshot_index: usize) -> String {
    format!("transient.energy_balance_residual.{snapshot_index}")
}

pub fn fea_transient_relative_energy_balance_error_field_id(snapshot_index: usize) -> String {
    format!("transient.relative_energy_balance_error.{snapshot_index}")
}

pub fn fea_nonlinear_displacement_field_id(increment_index: usize) -> String {
    format!("nonlinear.displacement.{increment_index}")
}

pub fn fea_nonlinear_rotation_field_id(increment_index: usize) -> String {
    format!("nonlinear.rotation.{increment_index}")
}

pub fn fea_nonlinear_von_mises_field_id(increment_index: usize) -> String {
    format!("nonlinear.von_mises.{increment_index}")
}

pub fn fea_nonlinear_plastic_strain_field_id(increment_index: usize) -> String {
    format!("nonlinear.plastic_strain.{increment_index}")
}

pub fn fea_nonlinear_equivalent_plastic_strain_field_id(increment_index: usize) -> String {
    format!("nonlinear.equivalent_plastic_strain.{increment_index}")
}

pub fn fea_nonlinear_contact_pressure_field_id(increment_index: usize) -> String {
    format!("nonlinear.contact_pressure.{increment_index}")
}

pub fn fea_nonlinear_contact_gap_field_id(increment_index: usize) -> String {
    format!("nonlinear.contact_gap.{increment_index}")
}

pub fn fea_nonlinear_load_factor_field_id(increment_index: usize) -> String {
    format!("nonlinear.load_factor.{increment_index}")
}

pub fn fea_nonlinear_residual_norm_field_id(increment_index: usize) -> String {
    format!("nonlinear.residual_norm.{increment_index}")
}

pub fn fea_thermo_mechanical_temperature_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.temperature.{step_index}")
}

pub fn fea_thermo_mechanical_thermal_strain_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.thermal_strain.{step_index}")
}

pub fn fea_thermo_mechanical_thermal_stress_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.thermal_stress.{step_index}")
}

pub fn fea_thermo_mechanical_displacement_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.displacement.{step_index}")
}

pub fn fea_thermo_mechanical_von_mises_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.von_mises.{step_index}")
}

pub fn fea_thermo_mechanical_coupling_residual_field_id(step_index: usize) -> String {
    format!("thermo_mechanical.coupling_residual.{step_index}")
}

pub const FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL: &str = "electro_thermal.electric_potential";
pub const FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD: &str = "electro_thermal.electric_field";
pub const FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY: &str = "electro_thermal.current_density";
pub const FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT: &str = "electro_thermal.joule_heat";

pub fn fea_electro_thermal_temperature_field_id(step_index: usize) -> String {
    format!("electro_thermal.temperature.{step_index}")
}

pub fn fea_electro_thermal_thermal_residual_field_id(step_index: usize) -> String {
    format!("electro_thermal.thermal_residual.{step_index}")
}

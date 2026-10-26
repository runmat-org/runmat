//! Dynamic buffers owned by one material. Inline coefficients live in the parent.
use super::*;
#[cfg(test)]
mod tests;
fn add(total: &mut u64, bytes: usize) -> Option<()> {
    *total = total.checked_add(u64::try_from(bytes).ok()?)?;
    Some(())
}
fn optional(total: &mut u64, value: &Option<String>) -> Option<()> {
    add(total, value.as_ref().map_or(0, String::capacity))
}
impl MaterialModel {
    /// Original dynamic buffer capacities, excluding this inline struct and allocator metadata.
    /// Sharing of a containing property inventory is accounted by its owner.
    pub fn owned_buffer_bytes(&self) -> Option<u64> {
        let Self {
            material_id,
            name,
            provenance,
            electrical,
            mass_density_kg_per_m3: _,
            elasticity,
            damping: _,
            thermal,
            acoustic,
            fluid: _,
            plastic,
        } = self;
        // Exhaustive inline law inventories make new coefficient-owner fields
        // require a storage review as well as the outer material contract.
        match elasticity {
            None => (),
            Some(MaterialElasticityModel::IsotropicLinear(MaterialMechanicalModel {
                youngs_modulus_pa: _,
                poisson_ratio: _,
            })) => (),
            Some(MaterialElasticityModel::OrthotropicLamina(MaterialOrthotropicLaminaModel {
                longitudinal_modulus_pa: _,
                transverse_modulus_pa: _,
                major_poisson_ratio: _,
                in_plane_shear_modulus_pa: _,
                transverse_shear_modulus_13_pa: _,
                transverse_shear_modulus_23_pa: _,
            })) => (),
        }
        if let Some(MaterialThermalModel {
            reference_temperature_k: _,
            modulus_temp_coeff_per_k: _,
            conductivity_w_per_mk: _,
            specific_heat_j_per_kgk: _,
            expansion_coefficient_per_k: _,
        }) = thermal
        {}
        if let Some(MaterialAcousticModel {
            speed_of_sound_m_per_s: _,
            damping_ratio: _,
        }) = acoustic
        {}
        if let Some(MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: _,
            isotropic_hardening_modulus_pa: _,
        }) = plastic
        {}
        let mut total = 0;
        add(&mut total, material_id.capacity())?;
        add(&mut total, name.capacity())?;
        if let Some(MaterialElectricalModel {
            conductivity_frequency_response,
            reference_temperature_k: _,
            conductivity_s_per_m: _,
            resistive_heating_coefficient: _,
            relative_permittivity: _,
            relative_permeability: _,
        }) = electrical
        {
            add(
                &mut total,
                conductivity_frequency_response
                    .capacity()
                    .checked_mul(std::mem::size_of::<ConductivityFrequencyPoint>())?,
            )?;
        }
        match provenance {
            MaterialProvenance::Authored => (),
            MaterialProvenance::ImportedCad(value) => {
                let ImportedCadMaterialProvenance {
                    geometry_module_path,
                    geometry_id,
                    source_material_id,
                    source_material_name,
                    source_entity_id: _,
                    persistent_id,
                    specification,
                    associations,
                    adopted_properties,
                } = value;
                for value in [
                    geometry_module_path,
                    geometry_id,
                    source_material_id,
                    source_material_name,
                ] {
                    add(&mut total, value.capacity())?;
                }
                optional(&mut total, persistent_id)?;
                optional(&mut total, specification)?;
                add(
                    &mut total,
                    associations
                        .capacity()
                        .checked_mul(std::mem::size_of::<ImportedCadMaterialAssociation>())?,
                )?;
                for ImportedCadMaterialAssociation {
                    association_id,
                    source_entity_id: _,
                    target,
                } in associations
                {
                    add(&mut total, association_id.capacity())?;
                    if let ImportedCadMaterialAssociationTarget::PersistentId(value) = target {
                        add(&mut total, value.capacity())?;
                    }
                }
                add(
                    &mut total,
                    adopted_properties
                        .capacity()
                        .checked_mul(std::mem::size_of::<ImportedCadMaterialProperty>())?,
                )?;
                for ImportedCadMaterialProperty {
                    property: _,
                    source_property_id,
                    source_name,
                    source_entity_id: _,
                    representation_entity_id: _,
                    value,
                    specification,
                    environment,
                } in adopted_properties
                {
                    add(&mut total, value.owned_buffer_bytes()?)?;
                    for value in [source_property_id, source_name] {
                        add(&mut total, value.capacity())?;
                    }
                    optional(&mut total, specification)?;
                    add(
                        &mut total,
                        environment
                            .capacity()
                            .checked_mul(std::mem::size_of::<ImportedCadEnvironmentCondition>())?,
                    )?;
                    for ImportedCadEnvironmentCondition { name, value } in environment {
                        add(&mut total, name.capacity())?;
                        add(&mut total, value.owned_buffer_bytes()?)?;
                    }
                }
            }
        }
        Some(total)
    }
}

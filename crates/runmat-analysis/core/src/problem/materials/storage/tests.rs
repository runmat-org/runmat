use super::*;

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Cleared conductivity-response storage retains its allocated capacity and remains
//   charged by the material owner. Typed component demand only; no allocator, complete-study peak
//   or physics claim.
#[test]
fn retained_material_buffers_include_unused_electrical_capacity() {
    let mut material = super::super::tests::density_only_material();
    let baseline = material.owned_buffer_bytes().unwrap();
    let mut response = Vec::with_capacity(128);
    response.push(ConductivityFrequencyPoint {
        frequency_hz: runmat_types::physical::Quantity::<runmat_types::physical::kind::Frequency,_>::new(1.0),
        conductivity_scale: 1.0,
        dispersive_loss_scale: None,
        relative_permittivity_scale: None,
        relative_permeability_scale: None,
    });
    response.clear();
    let allocated = response.capacity() * std::mem::size_of::<ConductivityFrequencyPoint>();
    material.electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(298.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        conductivity_frequency_response: response,
    });
    assert_eq!(
        material.owned_buffer_bytes(),
        Some(baseline + allocated as u64)
    );
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Imported CAD collection capacity and persistent-target strings are owned separately;
//   clearing entries releases strings while retaining collection storage. Typed component demand,
//   not measured peak memory or complete-study scalability qualification.
#[test]
fn retained_material_buffers_include_cad_collection_capacity_and_target_strings() {
    let mut material = super::super::tests::density_only_material();
    let baseline = material.owned_buffer_bytes().unwrap();
    let mut target = String::with_capacity(512);
    target.push_str("original-face");
    let target_capacity = target.capacity();
    let mut associations = Vec::with_capacity(64);
    associations.push(ImportedCadMaterialAssociation {
        association_id: String::new(),
        source_entity_id: 1,
        target: ImportedCadMaterialAssociationTarget::PersistentId(target),
    });
    let association_capacity =
        associations.capacity() * std::mem::size_of::<ImportedCadMaterialAssociation>();
    let properties = Vec::with_capacity(32);
    let property_capacity =
        properties.capacity() * std::mem::size_of::<ImportedCadMaterialProperty>();
    material.provenance = MaterialProvenance::ImportedCad(ImportedCadMaterialProvenance {
        geometry_module_path: String::new(),
        geometry_id: String::new(),
        source_material_id: String::new(),
        source_material_name: String::new(),
        source_entity_id: 1,
        persistent_id: None,
        specification: None,
        associations,
        adopted_properties: properties,
    });
    assert_eq!(
        material.owned_buffer_bytes(),
        Some(baseline + (association_capacity + property_capacity + target_capacity) as u64)
    );
    let MaterialProvenance::ImportedCad(provenance) = &mut material.provenance else {
        unreachable!()
    };
    provenance.associations.clear();
    assert_eq!(
        material.owned_buffer_bytes(),
        Some(baseline + (association_capacity + property_capacity) as u64)
    );
}

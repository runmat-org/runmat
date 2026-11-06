//! Retained compiled inventories: typed buffers, shared owners and map-node allowance.
use super::*;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
fn add(total: &mut u64, bytes: u64) -> Result<(), String> {
    *total = total
        .checked_add(bytes)
        .ok_or("property storage overflow")?;
    Ok(())
}
fn map_slots<K, V>(count: usize) -> Result<u64, String> {
    // B-tree nodes have eleven value slots and at least five occupied slots
    // outside the root. Three slots plus links per entry bound this structural
    // storage; the fixed contract allowance covers partially occupied roots.
    (count as u64)
        .checked_mul(3 * (std::mem::size_of::<(K, V)>() + 8 * std::mem::size_of::<usize>()) as u64)
        .ok_or_else(|| "property map storage overflow".into())
}
impl CompiledPropertyMap {
    /// Original retained storage, charged once while additional clones share it.
    /// This is a conservative structural demand, not measured allocator usage.
    pub fn retained_storage_bytes_estimate(&self, limits: CanonicalLimits) -> Result<u64, String> {
        let Self {
            cell_materials,
            identity,
            audit_digest: _,
            solid_materials,
            beam_properties,
            shell_properties,
            audit,
        } = self;
        let mut total = std::mem::size_of::<Self>() as u64
            + runmat_canonical_codec::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT;
        add(&mut total, identity.geometry_id.capacity() as u64)?;
        add(&mut total, (5 * 2 * std::mem::size_of::<usize>()) as u64)?;
        // Elements reference one shared owner per material: charge each distinct owner once
        // (its allocation and owned buffers); the per-element reference is in the map slot.
        let mut owners = std::collections::BTreeSet::new();
        for map in [cell_materials, solid_materials] {
            add(
                &mut total,
                map_slots::<StableDigest, CompiledElementMaterial>(map.len())?,
            )?;
            for CompiledElementMaterial {
                element_id: _,
                element_identity: _,
                material,
            } in map.values()
            {
                if owners.insert(Arc::as_ptr(material) as usize) {
                    add(
                        &mut total,
                        (std::mem::size_of::<MaterialModel>() + 2 * std::mem::size_of::<usize>())
                            as u64,
                    )?;
                    add(
                        &mut total,
                        material
                            .owned_buffer_bytes()
                            .ok_or("material storage overflow")?,
                    )?;
                }
            }
        }
        add(&mut total, map_slots::<usize, ()>(owners.len())?)?;
        // Beam/shell metadata has variable section/ply owners. Its conservative
        // shared projection remains additional to map slots, never a solid-map tree.
        add(
            &mut total,
            map_slots::<StableDigest, CompiledBeamProperty>(beam_properties.len())?,
        )?;
        add(
            &mut total,
            map_slots::<StableDigest, CompiledShellProperty>(shell_properties.len())?,
        )?;
        // Keys (digests) are charged by the map slots above; profile only the owned values,
        // since canonical profiles admit scalar map keys only.
        let sections = profile_serialized(
            &(
                beam_properties.values().collect::<Vec<_>>(),
                shell_properties.values().collect::<Vec<_>>(),
            ),
            limits,
        )
        .map_err(|error| error.to_string())?
        .materialization_bytes_without_envelope()
        .ok_or("section profile overflow")?;
        add(&mut total, sections)?;
        add(
            &mut total,
            (audit.capacity() as u64)
                .checked_mul(std::mem::size_of::<CompiledPropertyAudit>() as u64)
                .ok_or("audit storage overflow")?,
        )?;
        let mut ids = std::collections::BTreeSet::new();
        for CompiledPropertyAudit {
            element_identity: _,
            owner,
        } in audit.iter()
        {
            match owner {
                CompiledPropertyAuditOwner::MaterialAssignment {
                    assignment_index: _,
                    expected_material_id,
                    assigned_material_id,
                    confidence: _,
                } => {
                    // Elements of one assignment share its id owners: charge each once.
                    for id in [expected_material_id, assigned_material_id] {
                        if ids.insert(Arc::as_ptr(id) as *const u8 as usize) {
                            add(
                                &mut total,
                                (id.len() + 2 * std::mem::size_of::<usize>()) as u64,
                            )?;
                        }
                    }
                }
                CompiledPropertyAuditOwner::Laminate {
                    section_id,
                    ordered_ply_material_ids,
                } => {
                    add(&mut total, section_id.capacity() as u64)?;
                    add(
                        &mut total,
                        (ordered_ply_material_ids.capacity() as u64)
                            .checked_mul(std::mem::size_of::<String>() as u64)
                            .ok_or("ply storage overflow")?,
                    )?;
                    for id in ordered_ply_material_ids {
                        add(&mut total, id.capacity() as u64)?;
                    }
                }
            }
        }
        add(&mut total, map_slots::<usize, ()>(ids.len())?)?;
        if total > limits.maximum_encoded_bytes as u64 {
            return Err("property storage exceeds budget".into());
        }
        Ok(total)
    }
}

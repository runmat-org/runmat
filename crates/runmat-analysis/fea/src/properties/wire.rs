//! Compact persisted form of a compiled property map.
//!
//! Solid and cell inventories repeat one of a few materials across every element, so the wire
//! keeps a material table and per-element columns. Audit entries point at their element by
//! position in the sorted element inventory and at an owner table. Digests are lowercase hex.
//! Semantic and audit identities are computed from the typed maps, never from this encoding.
use super::*;
use serde::{de::Error as _, Deserializer, Serializer};

#[derive(Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ElementColumns {
    element_ids: Vec<u64>,
    identities: Vec<HexDigest>,
    materials: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditColumns {
    elements: Vec<u32>,
    owners: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    identity: CompiledPropertyMapIdentity,
    audit_digest: Digest,
    materials: Vec<MaterialModel>,
    solids: ElementColumns,
    #[serde(default, skip_serializing_if = "ElementColumns::is_empty")]
    cells: ElementColumns,
    beam_properties: Vec<CompiledBeamProperty>,
    shell_properties: Vec<CompiledShellProperty>,
    audit_owners: Vec<CompiledPropertyAuditOwner>,
    audit: AuditColumns,
}

impl ElementColumns {
    fn is_empty(&self) -> bool {
        self.element_ids.is_empty()
    }
}

struct HexDigest(StableDigest);

impl Serialize for HexDigest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut text = [0_u8; 64];
        for (index, byte) in self.0.bytes().iter().enumerate() {
            text[2 * index] = HEX[(byte >> 4) as usize];
            text[2 * index + 1] = HEX[(byte & 15) as usize];
        }
        serializer.serialize_str(std::str::from_utf8(&text).map_err(serde::ser::Error::custom)?)
    }
}

impl<'de> Deserialize<'de> for HexDigest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = std::borrow::Cow::<'de, str>::deserialize(deserializer)?;
        let text = text.as_bytes();
        if text.len() != 64 {
            return Err(D::Error::custom(
                "element digest must be 64 lowercase hex digits",
            ));
        }
        let nibble = |c: u8| match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            _ => Err(D::Error::custom("element digest must be lowercase hex")),
        };
        let mut bytes = [0_u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (nibble(text[2 * index])? << 4) | nibble(text[2 * index + 1])?;
        }
        Ok(Self(StableDigest::from_bytes(bytes)))
    }
}

fn material_index(table: &mut Vec<MaterialModel>, material: &MaterialModel) -> Result<u32, String> {
    let index = match table.iter().position(|known| known == material) {
        Some(index) => index,
        None => {
            table.push(material.clone());
            table.len() - 1
        }
    };
    u32::try_from(index).map_err(|_| "compiled property material table overflow".into())
}

fn columns(
    entries: &BTreeMap<StableDigest, CompiledElementMaterial>,
    table: &mut Vec<MaterialModel>,
) -> Result<ElementColumns, String> {
    let mut columns = ElementColumns {
        element_ids: Vec::with_capacity(entries.len()),
        identities: Vec::with_capacity(entries.len()),
        materials: Vec::with_capacity(entries.len()),
    };
    for entry in entries.values() {
        columns.element_ids.push(entry.element_id);
        columns.identities.push(HexDigest(entry.element_identity));
        columns
            .materials
            .push(material_index(table, &entry.material)?);
    }
    Ok(columns)
}

fn rows(
    columns: ElementColumns,
    table: &[Arc<MaterialModel>],
    role: &str,
) -> Result<BTreeMap<StableDigest, CompiledElementMaterial>, String> {
    let count = columns.element_ids.len();
    if columns.identities.len() != count || columns.materials.len() != count {
        return Err(format!("compiled {role} property columns differ in length"));
    }
    let mut map = BTreeMap::new();
    for ((element_id, identity), material) in columns
        .element_ids
        .into_iter()
        .zip(columns.identities)
        .zip(columns.materials)
    {
        let material = table
            .get(material as usize)
            .ok_or_else(|| format!("compiled {role} property names a missing material"))?
            .clone();
        let entry = CompiledElementMaterial {
            element_id,
            element_identity: identity.0,
            material,
        };
        if map.insert(identity.0, entry).is_some() {
            return Err("compiled property map contains duplicate element identities".into());
        }
    }
    Ok(map)
}

/// Sorted element inventory in role order; audit positions index this list.
fn element_order(map: &CompiledPropertyMap) -> Vec<StableDigest> {
    map.solid_entries()
        .keys()
        .chain(map.cell_entries().keys())
        .chain(map.beam_entries().keys())
        .chain(map.shell_entries().keys())
        .copied()
        .collect()
}

pub(super) fn serialize<S: Serializer>(
    map: &CompiledPropertyMap,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::Error;
    let mut materials = Vec::new();
    let solids = columns(map.solid_entries(), &mut materials).map_err(S::Error::custom)?;
    let cells = columns(map.cell_entries(), &mut materials).map_err(S::Error::custom)?;
    let mut order = element_order(map)
        .into_iter()
        .enumerate()
        .map(|(position, identity)| (identity, position))
        .collect::<Vec<_>>();
    order.sort_unstable();
    let mut audit_owners: Vec<CompiledPropertyAuditOwner> = Vec::new();
    let mut audit = AuditColumns {
        elements: Vec::with_capacity(map.audit().len()),
        owners: Vec::with_capacity(map.audit().len()),
    };
    for entry in map.audit() {
        let position = order
            .binary_search_by(|probe| probe.0.cmp(&entry.element_identity))
            .map(|index| order[index].1)
            .map_err(|_| S::Error::custom("compiled property audit names an unknown element"))?;
        let owner = match audit_owners.iter().position(|known| *known == entry.owner) {
            Some(index) => index,
            None => {
                audit_owners.push(entry.owner.clone());
                audit_owners.len() - 1
            }
        };
        audit.elements.push(
            u32::try_from(position).map_err(|_| S::Error::custom("audit position overflow"))?,
        );
        audit
            .owners
            .push(u32::try_from(owner).map_err(|_| S::Error::custom("audit owner overflow"))?);
    }
    Wire {
        identity: map.identity.clone(),
        audit_digest: map.audit_digest,
        materials,
        solids,
        cells,
        beam_properties: map.beam_entries().values().cloned().collect(),
        shell_properties: map.shell_entries().values().cloned().collect(),
        audit_owners,
        audit,
    }
    .serialize(serializer)
}

pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<CompiledPropertyMap, D::Error> {
    let wire = Wire::deserialize(deserializer)?;
    from_wire(wire).map_err(D::Error::custom)
}

fn keyed<T>(
    values: Vec<T>,
    identity: impl Fn(&T) -> StableDigest,
) -> Result<BTreeMap<StableDigest, T>, String> {
    let mut map = BTreeMap::new();
    for value in values {
        if map.insert(identity(&value), value).is_some() {
            return Err("compiled property map contains duplicate element identities".into());
        }
    }
    Ok(map)
}

fn from_wire(wire: Wire) -> Result<CompiledPropertyMap, String> {
    // One shared owner per table entry; every element row references it.
    let table = wire.materials.into_iter().map(Arc::new).collect::<Vec<_>>();
    let solid_materials = rows(wire.solids, &table, "solid")?;
    let cell_materials = rows(wire.cells, &table, "cell")?;
    let beam_properties = keyed(wire.beam_properties, |entry| entry.element_identity)?;
    let shell_properties = keyed(wire.shell_properties, |entry| entry.element_identity)?;
    if wire.audit.elements.len() != wire.audit.owners.len() {
        return Err("compiled property audit columns differ in length".into());
    }
    let order = solid_materials
        .keys()
        .chain(cell_materials.keys())
        .chain(beam_properties.keys())
        .chain(shell_properties.keys())
        .copied()
        .collect::<Vec<_>>();
    let audit = wire
        .audit
        .elements
        .iter()
        .zip(&wire.audit.owners)
        .map(|(element, owner)| {
            Ok(CompiledPropertyAudit {
                element_identity: *order
                    .get(*element as usize)
                    .ok_or("compiled property audit names an unknown element")?,
                owner: wire
                    .audit_owners
                    .get(*owner as usize)
                    .ok_or("compiled property audit names an unknown owner")?
                    .clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let compiled = CompiledPropertyMap {
        identity: wire.identity,
        audit_digest: wire.audit_digest,
        solid_materials: Arc::new(solid_materials),
        cell_materials: Arc::new(cell_materials),
        beam_properties: Arc::new(beam_properties),
        shell_properties: Arc::new(shell_properties),
        audit: Arc::new(audit),
    };
    compiled.validate_integrity()?;
    Ok(compiled)
}

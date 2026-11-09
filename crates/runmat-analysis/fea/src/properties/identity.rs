use std::collections::BTreeMap;

use runmat_analysis_core::{
    BeamSectionModel, HomogeneousShellSectionModel, LaminatePlyModel, LaminateShellSectionModel,
    SolverMaterialModel,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::{Digest, DigestStream};
use runmat_meshing_core::StableDigest;
use serde::{Serialize, Serializer};

use super::contract::CompiledPropertySource;
use super::{
    CompiledBeamProperty, CompiledElementMaterial, CompiledLaminatePly, CompiledPropertyAudit,
    CompiledPropertyMap, CompiledShellProperty, CompiledShellPropertyOwner,
};

// v2: element collections enter the identity as streamed sequence digests, so the identity of
// a mesh-sized property map never materializes one mesh-sized encoding (v1 needed a single
// buffer of ~390 bytes per element, which exceeded its ceiling at 1.4M Tet10 elements).
const SEMANTIC_PREFIX: &[u8] = b"runmat-fea-compiled-properties/v2\0";
const AUDIT_PREFIX: &[u8] = b"runmat-fea-compiled-property-audit/v2\0";
const SEMANTIC_ELEMENT_PREFIX: &[u8] = b"runmat-fea-compiled-property-element/v2\0";
const AUDIT_ELEMENT_PREFIX: &[u8] = b"runmat-fea-compiled-property-audit-element/v2\0";
const SEQUENCE_DIGEST_PREFIX: &[u8] = b"runmat-fea-compiled-property-sequence/v2\0";
/// Header contracts hold scalars, digests and a fixed number of sequence digests: their size
/// does not grow with the mesh.
pub(super) const HEADER_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024, 64, 4096, 64);
/// One element's semantics (material, section, laminate plies). Element count never enters
/// this bound; every element is encoded on its own.
const ELEMENT_LIMITS: CanonicalLimits =
    CanonicalLimits::content_bounded(64 * 1024 * 1024, 4096, 64);

/// Identity of an ordered element sequence: its length and the SHA-256 of the
/// length-delimited canonical encoding of each element, hashed as one stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct SequenceDigest {
    count: u64,
    digest: Digest,
}

fn sequence_digest<T: Serialize>(
    element_prefix: &[u8],
    domain: &str,
    elements: impl Iterator<Item = T>,
) -> Result<SequenceDigest, String> {
    let mut hash = DigestStream::new();
    hash.update(SEQUENCE_DIGEST_PREFIX);
    hash.update(&(domain.len() as u64).to_le_bytes());
    hash.update(domain.as_bytes());
    let mut count = 0u64;
    for element in elements {
        let bytes = encode_contract(element_prefix, domain, &element, ELEMENT_LIMITS)
            .map_err(|error| error.to_string())?;
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
        count += 1;
    }
    Ok(SequenceDigest {
        count,
        digest: hash.finish(),
    })
}

#[derive(Serialize)]
struct SemanticContract<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    cells: Option<SequenceDigest>,
    geometry_id: &'a str,
    geometry_revision: u32,
    mesh_digest: Option<StableDigest>,
    explicit_topology_digest: Option<Digest>,
    solids: SequenceDigest,
    beams: SequenceDigest,
    shells: SequenceDigest,
}

#[derive(Serialize)]
struct SolidSemanticElement<'a> {
    element_id: u64,
    element_identity: StableDigest,
    material: SolverMaterialModel<'a>,
}

#[derive(Serialize)]
struct BeamSemanticElement<'a> {
    element_id: &'a str,
    element_identity: StableDigest,
    material: SolverMaterialModel<'a>,
    section: &'a BeamSectionModel,
    reference_axis: [f64; 3],
}

#[derive(Serialize)]
struct ShellSemanticElement<'a> {
    element_id: &'a str,
    element_identity: StableDigest,
    reference_axis: [f64; 3],
    owner: ShellSemanticOwner<'a>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum ShellSemanticOwner<'a> {
    Homogeneous {
        material: SolverMaterialModel<'a>,
        section: &'a HomogeneousShellSectionModel,
    },
    Laminate {
        section: &'a LaminateShellSectionModel,
        #[serde(serialize_with = "serialize_plies")]
        plies: &'a [CompiledLaminatePly],
    },
}

#[derive(Serialize)]
struct LaminatePlySemantic<'a> {
    definition: &'a LaminatePlyModel,
    material: SolverMaterialModel<'a>,
}

#[derive(Serialize)]
struct AuditContract {
    #[serde(skip_serializing_if = "Option::is_none")]
    cells: Option<SequenceDigest>,
    semantic_digest: Digest,
    solids: SequenceDigest,
    beams: SequenceDigest,
    shells: SequenceDigest,
    assignments: SequenceDigest,
}

pub(super) fn validate(compiled: &CompiledPropertyMap) -> Result<(), String> {
    validate_structure(compiled)?;
    validate_digests(compiled)
}

/// Key, role and audit-coverage checks; digests are checked separately because a map whose
/// digests were just derived from these exact contents need not encode them a second time.
pub(super) fn validate_structure(compiled: &CompiledPropertyMap) -> Result<(), String> {
    if compiled.identity.geometry_id.trim().is_empty() {
        return Err("compiled property map geometry id is empty".into());
    }
    validate_entry_keys(
        compiled.solid_entries(),
        |entry| entry.element_identity,
        "solid",
    )?;
    validate_entry_keys(
        compiled.cell_entries(),
        |entry| entry.element_identity,
        "cell",
    )?;
    validate_entry_keys(
        compiled.beam_entries(),
        |entry| entry.element_identity,
        "beam",
    )?;
    validate_entry_keys(
        compiled.shell_entries(),
        |entry| entry.element_identity,
        "shell",
    )?;
    let mut element_identities = compiled
        .solid_entries()
        .keys()
        .chain(compiled.cell_entries().keys())
        .chain(compiled.beam_entries().keys())
        .chain(compiled.shell_entries().keys())
        .copied()
        .collect::<Vec<_>>();
    element_identities.sort_unstable();
    if element_identities.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("compiled property map assigns one element identity to multiple roles".into());
    }
    let mut audit_identities = compiled
        .audit()
        .iter()
        .map(|entry| entry.element_identity)
        .collect::<Vec<_>>();
    audit_identities.sort_unstable();
    if audit_identities != element_identities {
        return Err("compiled property audit does not cover the exact compiled element set".into());
    }
    drop(audit_identities);
    drop(element_identities);
    Ok(())
}

fn validate_digests(compiled: &CompiledPropertyMap) -> Result<(), String> {
    let semantic = semantic_digest(
        &CompiledPropertySource {
            geometry_id: compiled.identity.geometry_id.clone(),
            geometry_revision: compiled.identity.geometry_revision,
            mesh_digest: compiled.identity.mesh_digest,
            explicit_topology_digest: compiled.identity.explicit_topology_digest,
        },
        compiled.solid_entries(),
        compiled.cell_entries(),
        compiled.beam_entries(),
        compiled.shell_entries(),
    )?;
    if semantic != compiled.identity.semantic_digest {
        return Err("compiled property semantic digest does not match its contents".into());
    }
    if audit_digest(
        semantic,
        compiled.solid_entries(),
        compiled.cell_entries(),
        compiled.beam_entries(),
        compiled.shell_entries(),
        compiled.audit(),
    )? != compiled.audit_digest
    {
        return Err("compiled property audit digest does not match its contents".into());
    }
    Ok(())
}

fn validate_entry_keys<T>(
    entries: &BTreeMap<StableDigest, T>,
    identity: impl Fn(&T) -> StableDigest,
    role: &str,
) -> Result<(), String> {
    if entries.iter().any(|(key, entry)| *key != identity(entry)) {
        Err(format!(
            "compiled {role} property map key differs from its element identity"
        ))
    } else {
        Ok(())
    }
}

pub(super) fn semantic_digest(
    source: &CompiledPropertySource,
    solid_materials: &BTreeMap<StableDigest, CompiledElementMaterial>,
    cell_materials: &BTreeMap<StableDigest, CompiledElementMaterial>,
    beam_properties: &BTreeMap<StableDigest, CompiledBeamProperty>,
    shell_properties: &BTreeMap<StableDigest, CompiledShellProperty>,
) -> Result<Digest, String> {
    const DOMAIN: &str = "compiled-property-semantics";
    let contract = SemanticContract {
        cells: (!cell_materials.is_empty())
            .then(|| {
                sequence_digest(
                    SEMANTIC_ELEMENT_PREFIX,
                    "compiled-property-semantics.cells",
                    cell_materials.values().map(solid_semantics),
                )
            })
            .transpose()?,
        geometry_id: &source.geometry_id,
        geometry_revision: source.geometry_revision,
        mesh_digest: source.mesh_digest,
        explicit_topology_digest: source.explicit_topology_digest,
        solids: sequence_digest(
            SEMANTIC_ELEMENT_PREFIX,
            "compiled-property-semantics.solids",
            solid_materials.values().map(solid_semantics),
        )?,
        beams: sequence_digest(
            SEMANTIC_ELEMENT_PREFIX,
            "compiled-property-semantics.beams",
            beam_properties.values().map(beam_semantics),
        )?,
        shells: sequence_digest(
            SEMANTIC_ELEMENT_PREFIX,
            "compiled-property-semantics.shells",
            shell_properties.values().map(shell_semantics),
        )?,
    };
    encode_contract(SEMANTIC_PREFIX, DOMAIN, &contract, HEADER_LIMITS)
        .map(Digest::sha256)
        .map_err(|error| error.to_string())
}

pub(super) fn audit_digest(
    semantic_digest: Digest,
    solid_materials: &BTreeMap<StableDigest, CompiledElementMaterial>,
    cell_materials: &BTreeMap<StableDigest, CompiledElementMaterial>,
    beam_properties: &BTreeMap<StableDigest, CompiledBeamProperty>,
    shell_properties: &BTreeMap<StableDigest, CompiledShellProperty>,
    audit: &[CompiledPropertyAudit],
) -> Result<Digest, String> {
    const DOMAIN: &str = "compiled-property-audit";
    let contract = AuditContract {
        cells: (!cell_materials.is_empty())
            .then(|| {
                sequence_digest(
                    AUDIT_ELEMENT_PREFIX,
                    "compiled-property-audit.cells",
                    cell_materials.values(),
                )
            })
            .transpose()?,
        semantic_digest,
        solids: sequence_digest(
            AUDIT_ELEMENT_PREFIX,
            "compiled-property-audit.solids",
            solid_materials.values(),
        )?,
        beams: sequence_digest(
            AUDIT_ELEMENT_PREFIX,
            "compiled-property-audit.beams",
            beam_properties.values(),
        )?,
        shells: sequence_digest(
            AUDIT_ELEMENT_PREFIX,
            "compiled-property-audit.shells",
            shell_properties.values(),
        )?,
        assignments: sequence_digest(
            AUDIT_ELEMENT_PREFIX,
            "compiled-property-audit.assignments",
            audit.iter(),
        )?,
    };
    encode_contract(AUDIT_PREFIX, DOMAIN, &contract, HEADER_LIMITS)
        .map(Digest::sha256)
        .map_err(|error| error.to_string())
}

fn solid_semantics(entry: &CompiledElementMaterial) -> SolidSemanticElement<'_> {
    SolidSemanticElement {
        element_id: entry.element_id,
        element_identity: entry.element_identity,
        material: entry.material.solver_identity_contract(),
    }
}

fn beam_semantics(entry: &CompiledBeamProperty) -> BeamSemanticElement<'_> {
    BeamSemanticElement {
        element_id: &entry.element_id,
        element_identity: entry.element_identity,
        material: entry.material.solver_identity_contract(),
        section: &entry.section,
        reference_axis: entry.reference_axis,
    }
}

fn shell_semantics(entry: &CompiledShellProperty) -> ShellSemanticElement<'_> {
    let owner = match &entry.owner {
        CompiledShellPropertyOwner::Homogeneous { material, section } => {
            ShellSemanticOwner::Homogeneous {
                material: material.solver_identity_contract(),
                section,
            }
        }
        CompiledShellPropertyOwner::Laminate { section, plies } => {
            ShellSemanticOwner::Laminate { section, plies }
        }
    };
    ShellSemanticElement {
        element_id: &entry.element_id,
        element_identity: entry.element_identity,
        reference_axis: entry.reference_axis,
        owner,
    }
}

fn serialize_plies<S: Serializer>(
    plies: &[CompiledLaminatePly],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(plies.iter().map(|ply| LaminatePlySemantic {
        definition: &ply.definition,
        material: ply.material.solver_identity_contract(),
    }))
}

#[cfg(test)]
pub(super) mod tests;

use std::collections::BTreeMap;

use runmat_analysis_core::{
    BeamSectionModel, HomogeneousShellSectionModel, LaminatePlyModel, LaminateShellSectionModel,
    SolverMaterialModel,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::Serialize;

use super::contract::CompiledPropertySource;
use super::{
    CompiledBeamProperty, CompiledElementMaterial, CompiledPropertyAudit, CompiledPropertyMap,
    CompiledShellProperty, CompiledShellPropertyOwner,
};

const SEMANTIC_PREFIX: &[u8] = b"runmat-fea-compiled-properties/v1\0";
const AUDIT_PREFIX: &[u8] = b"runmat-fea-compiled-property-audit/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 4096, 64);

#[derive(Serialize)]
struct SemanticContract<'a> {
    geometry_id: &'a str,
    geometry_revision: u32,
    mesh_digest: Option<StableDigest>,
    explicit_topology_digest: Option<Digest>,
    solids: Vec<SolidSemanticElement<'a>>,
    beams: Vec<BeamSemanticElement<'a>>,
    shells: Vec<ShellSemanticElement<'a>>,
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
        plies: Vec<LaminatePlySemantic<'a>>,
    },
}

#[derive(Serialize)]
struct LaminatePlySemantic<'a> {
    definition: &'a LaminatePlyModel,
    material: SolverMaterialModel<'a>,
}

#[derive(Serialize)]
struct AuditContract<'a> {
    semantic_digest: Digest,
    solids: Vec<&'a CompiledElementMaterial>,
    beams: Vec<&'a CompiledBeamProperty>,
    shells: Vec<&'a CompiledShellProperty>,
    assignments: &'a [CompiledPropertyAudit],
}

pub(super) fn validate(compiled: &CompiledPropertyMap) -> Result<(), String> {
    if compiled.identity.geometry_id.trim().is_empty() {
        return Err("compiled property map geometry id is empty".into());
    }
    validate_entry_keys(
        compiled.solid_entries(),
        |entry| entry.element_identity,
        "solid",
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
        .chain(compiled.beam_entries().keys())
        .chain(compiled.shell_entries().keys())
        .copied()
        .collect::<Vec<_>>();
    element_identities.sort();
    if element_identities.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("compiled property map assigns one element identity to multiple roles".into());
    }
    let mut audit_identities = compiled
        .audit()
        .iter()
        .map(|entry| entry.element_identity)
        .collect::<Vec<_>>();
    audit_identities.sort();
    if audit_identities != element_identities {
        return Err("compiled property audit does not cover the exact compiled element set".into());
    }
    let semantic = semantic_digest(
        &CompiledPropertySource {
            geometry_id: compiled.identity.geometry_id.clone(),
            geometry_revision: compiled.identity.geometry_revision,
            mesh_digest: compiled.identity.mesh_digest,
            explicit_topology_digest: compiled.identity.explicit_topology_digest,
        },
        compiled.solid_entries(),
        compiled.beam_entries(),
        compiled.shell_entries(),
    )?;
    if semantic != compiled.identity.semantic_digest {
        return Err("compiled property semantic digest does not match its contents".into());
    }
    if audit_digest(
        semantic,
        compiled.solid_entries(),
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
    beam_properties: &BTreeMap<StableDigest, CompiledBeamProperty>,
    shell_properties: &BTreeMap<StableDigest, CompiledShellProperty>,
) -> Result<Digest, String> {
    encode_contract(
        SEMANTIC_PREFIX,
        "compiled-property-semantics",
        &SemanticContract {
            geometry_id: &source.geometry_id,
            geometry_revision: source.geometry_revision,
            mesh_digest: source.mesh_digest,
            explicit_topology_digest: source.explicit_topology_digest,
            solids: solid_materials
                .values()
                .map(|entry| SolidSemanticElement {
                    element_id: entry.element_id,
                    element_identity: entry.element_identity,
                    material: entry.material.solver_identity_contract(),
                })
                .collect(),
            beams: beam_properties
                .values()
                .map(|entry| BeamSemanticElement {
                    element_id: &entry.element_id,
                    element_identity: entry.element_identity,
                    material: entry.material.solver_identity_contract(),
                    section: &entry.section,
                    reference_axis: entry.reference_axis,
                })
                .collect(),
            shells: shell_properties.values().map(shell_semantics).collect(),
        },
        LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| error.to_string())
}

pub(super) fn audit_digest(
    semantic_digest: Digest,
    solid_materials: &BTreeMap<StableDigest, CompiledElementMaterial>,
    beam_properties: &BTreeMap<StableDigest, CompiledBeamProperty>,
    shell_properties: &BTreeMap<StableDigest, CompiledShellProperty>,
    audit: &[CompiledPropertyAudit],
) -> Result<Digest, String> {
    encode_contract(
        AUDIT_PREFIX,
        "compiled-property-audit",
        &AuditContract {
            semantic_digest,
            solids: solid_materials.values().collect(),
            beams: beam_properties.values().collect(),
            shells: shell_properties.values().collect(),
            assignments: audit,
        },
        LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| error.to_string())
}

fn shell_semantics(entry: &CompiledShellProperty) -> ShellSemanticElement<'_> {
    let owner = match &entry.owner {
        CompiledShellPropertyOwner::Homogeneous { material, section } => {
            ShellSemanticOwner::Homogeneous {
                material: material.solver_identity_contract(),
                section,
            }
        }
        CompiledShellPropertyOwner::Laminate { section, plies } => ShellSemanticOwner::Laminate {
            section,
            plies: plies
                .iter()
                .map(|ply| LaminatePlySemantic {
                    definition: &ply.definition,
                    material: ply.material.solver_identity_contract(),
                })
                .collect(),
        },
    };
    ShellSemanticElement {
        element_id: &entry.element_id,
        element_identity: entry.element_identity,
        reference_axis: entry.reference_axis,
        owner,
    }
}

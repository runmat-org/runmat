use std::collections::BTreeSet;

use runmat_builtins::{
    builtin_catalog_entries, BuiltinAcceleratorPolicy, BuiltinCatalogEntry, BuiltinFusionPolicy,
    BuiltinResidencyPolicy,
};

use super::super::schema::{
    GpuSpecRecord, MigrationFinding, MigrationFindingAffected, MigrationFindingCode,
};

pub(super) fn validate(findings: &mut Vec<MigrationFinding>, gpu_specs: &[GpuSpecRecord]) {
    let gpu_names = gpu_specs
        .iter()
        .map(|spec| spec.key)
        .collect::<BTreeSet<_>>();
    for entry in builtin_catalog_entries() {
        let name = entry.identity.name;
        if entry.placement.accelerator == BuiltinAcceleratorPolicy::Required
            && !gpu_names.contains(name)
        {
            findings.push(finding(
                name,
                "required accelerator policy has no compiled GPU specification",
            ));
        }
        if entry.placement.accelerator == BuiltinAcceleratorPolicy::Forbidden
            && entry.placement.residency == BuiltinResidencyPolicy::ProduceResident
        {
            findings.push(finding(
                name,
                "forbidden accelerator policy contradicts newly produced device-resident output policy",
            ));
        }
        let executable = has_executable_fusion(entry);
        match (entry.placement.fusion, executable) {
            (BuiltinFusionPolicy::Candidate, false) => findings.push(finding(
                name,
                "fusion candidate has no binding-owned executable fusion lowering",
            )),
            (BuiltinFusionPolicy::Never | BuiltinFusionPolicy::Boundary, true) => {
                findings.push(finding(
                    name,
                    "non-candidate fusion policy has a binding-owned executable fusion lowering",
                ));
            }
            (BuiltinFusionPolicy::Candidate, true)
            | (BuiltinFusionPolicy::Never | BuiltinFusionPolicy::Boundary, false) => {}
        }
    }
}

fn has_executable_fusion(entry: &BuiltinCatalogEntry) -> bool {
    use crate::builtins::common::spec::canonical_fusion_declaration;
    // Use the same exact implementation/binding join as execution, not a
    // textual spec key. A raw reduction shader is not a typed reduction route;
    // conversely structural graph operations need no scalar shader template.
    entry.bindings.iter().any(|binding| {
        canonical_fusion_declaration(entry.binding_identity(binding)).is_some_and(|declaration| {
            declaration.spec.elementwise.is_some()
                || declaration.structural.is_some()
                || declaration.epilogue.is_some()
                || declaration.reduction.is_some()
        })
    })
}

fn finding(name: &'static str, message: &str) -> MigrationFinding {
    MigrationFinding {
        code: MigrationFindingCode::PlacementContractMismatch,
        source: "placement_contract",
        affected: MigrationFindingAffected::Identity {
            identity: runmat_builtins::BuiltinCatalogIdentity { name },
        },
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_binding_lowerings_distinguish_structural_calls_and_dormant_templates() {
        assert!(has_executable_fusion(
            &runmat_builtins::MTIMES_CATALOG_ENTRY
        ));
        for entry in [
            &runmat_builtins::INT8_CATALOG_ENTRY,
            &runmat_builtins::MLDIVIDE_CATALOG_ENTRY,
            &runmat_builtins::MRDIVIDE_CATALOG_ENTRY,
            &runmat_builtins::MPOWER_CATALOG_ENTRY,
            &runmat_builtins::PROD_CATALOG_ENTRY,
            &runmat_builtins::NNZ_CATALOG_ENTRY,
            &runmat_builtins::ALL_CATALOG_ENTRY,
            &runmat_builtins::ANY_CATALOG_ENTRY,
        ] {
            assert!(!has_executable_fusion(entry), "{}", entry.identity.name);
            assert_eq!(entry.placement.fusion, BuiltinFusionPolicy::Boundary);
        }
        for entry in [
            &runmat_builtins::PROD_CATALOG_ENTRY,
            &runmat_builtins::NNZ_CATALOG_ENTRY,
            &runmat_builtins::ALL_CATALOG_ENTRY,
            &runmat_builtins::ANY_CATALOG_ENTRY,
        ] {
            let declaration = crate::builtins::common::spec::canonical_fusion_declaration(
                entry.binding_identity(&entry.bindings[0]),
            )
            .expect("retained canonical shader declaration");
            assert!(declaration.spec.reduction.is_some());
            assert!(declaration.reduction.is_none());
        }
        for entry in [
            &runmat_builtins::DEAL_CATALOG_ENTRY,
            &runmat_builtins::NAMEDARGS2CELL_CATALOG_ENTRY,
        ] {
            assert_eq!(
                entry.placement.accelerator,
                BuiltinAcceleratorPolicy::Forbidden
            );
            assert_eq!(
                entry.placement.residency,
                BuiltinResidencyPolicy::PreserveInputs
            );
        }
    }
}

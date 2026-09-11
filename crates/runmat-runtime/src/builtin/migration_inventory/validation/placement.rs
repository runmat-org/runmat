use std::collections::BTreeSet;

use runmat_builtins::{
    builtin_catalog_entries, BuiltinAcceleratorPolicy, BuiltinFusionPolicy, BuiltinResidencyPolicy,
};

use super::super::schema::{
    FusionSpecRecord, GpuSpecRecord, MigrationFinding, MigrationFindingCode,
};

pub(super) fn validate(
    findings: &mut Vec<MigrationFinding>,
    gpu_specs: &[GpuSpecRecord],
    fusion_specs: &[FusionSpecRecord],
) {
    let gpu_names = gpu_specs
        .iter()
        .map(|spec| spec.key)
        .collect::<BTreeSet<_>>();
    let active_fusion_names = fusion_specs
        .iter()
        .filter(|spec| spec.elementwise.is_some() || spec.reduction.is_some())
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
            && matches!(
                entry.placement.residency,
                BuiltinResidencyPolicy::PreserveInputs | BuiltinResidencyPolicy::ProduceResident
            )
        {
            findings.push(finding(
                name,
                "forbidden accelerator policy contradicts device-resident output policy",
            ));
        }
        match entry.placement.fusion {
            BuiltinFusionPolicy::Candidate if !active_fusion_names.contains(name) => findings.push(
                finding(name, "fusion candidate has no executable fusion template"),
            ),
            BuiltinFusionPolicy::Never | BuiltinFusionPolicy::Boundary
                if active_fusion_names.contains(name) =>
            {
                findings.push(finding(
                    name,
                    "non-candidate fusion policy has an executable fusion template",
                ));
            }
            _ => {}
        }
    }
}

fn finding(name: &str, message: &str) -> MigrationFinding {
    MigrationFinding {
        code: MigrationFindingCode::PlacementContractMismatch,
        source: "placement_contract",
        identity: name.to_owned(),
        message: message.to_owned(),
    }
}

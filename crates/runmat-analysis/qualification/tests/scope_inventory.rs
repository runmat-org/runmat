//! The scope inventory must map every capability the sources claim to obligations that exist.
use runmat_analysis_qualification::{requirements, scope};

// witness: FEA-RELEASE-REGISTER-001/scope_inventory_complete @ shared_infrastructure/mesh_independent/source_check;
//   boundary: Parses the checked-in inventory, rejects unmapped rows, duplicate ids and mappings to
//   absent requirements or obligations; does not judge whether a mapping is sufficient.
#[test]
fn every_scope_row_maps_to_existing_obligations_or_an_explicit_exclusion() {
    let rows = scope::rows().unwrap_or_else(|p| panic!("{p:?}"));
    assert!(
        rows.len() > 200,
        "inventory unexpectedly small: {}",
        rows.len()
    );
    let registry: Vec<_> = requirements().collect();
    let problems = scope::validate(&rows, &registry);
    assert!(
        problems.is_empty(),
        "{} scope problems:\n{}",
        problems.len(),
        problems
            .iter()
            .take(40)
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn every_requirement_is_supported_by_a_scope_row_or_listed_here() {
    let rows = scope::rows().unwrap();
    let mut referenced = std::collections::BTreeSet::new();
    for row in &rows {
        for map in &row.maps {
            if let scope::Mapping::Obligation { requirement, .. } = map {
                referenced.insert(requirement.clone());
            }
        }
    }
    let unsupported: Vec<_> = requirements()
        .map(|r| r.id)
        .filter(|id| !referenced.contains(*id))
        .collect();
    assert!(
        unsupported.is_empty(),
        "register entries no current source supports (map them or remove them): {unsupported:?}"
    );
}

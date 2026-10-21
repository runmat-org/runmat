use crate::algebra::{ContributionDomain, ContributionId, ContributionProvenance, OperatorRole};

pub(super) fn solid_operator_provenance(role: OperatorRole) -> ContributionProvenance {
    ContributionProvenance {
        domain: ContributionDomain::Structural,
        contribution_id: ContributionId(
            match role {
                OperatorRole::Stiffness => "solid_element_stiffness",
                OperatorRole::GeometricStiffness => "solid_element_geometric_stiffness",
                OperatorRole::Mass => "solid_element_consistent_mass",
                OperatorRole::Damping => "solid_element_rayleigh_damping",
                _ => "solid_element_operator",
            }
            .to_string(),
        ),
    }
}

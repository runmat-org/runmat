//! Compiled joins, not name-based runtime authorization. The six physical
//! owners scope L05 without maintaining a parallel list of builtin identities.
use super::*;
use runmat_builtins::{
    builtin_catalog_primary_entry_by_name, builtin_implementation_provenance, BuiltinFusionPolicy,
    BuiltinImplementationAuthority, BuiltinImplementationProvenance,
};
use std::collections::BTreeSet;

fn in_numeric_batch(path: &str) -> bool {
    let Some(path) = path.strip_prefix("crate::builtins::math::") else {
        return false;
    };
    matches!(
        path.split("::").next(),
        Some(
            "bitwise"
                | "discrete"
                | "elementwise"
                | "integer_division"
                | "rounding"
                | "trigonometry"
        )
    )
}

fn exact_owner<'a>(
    implementations: &'a [&BuiltinImplementationProvenance],
    name: &str,
    path: &str,
) -> Option<&'a BuiltinImplementationProvenance> {
    let mut owners = implementations.iter().copied().filter(|implementation| {
        implementation.authority == BuiltinImplementationAuthority::CanonicalBinding
            && implementation.name == name
            && implementation.builtin_path == path
    });
    let owner = owners.next()?;
    owners.next().is_none().then_some(owner)
}

#[test]
fn numeric_batch_compiled_gpu_and_fusion_declarations_join_exact_owners() {
    let implementations = builtin_implementation_provenance();
    let owners = implementations
        .iter()
        .copied()
        .filter(|implementation| {
            implementation.authority == BuiltinImplementationAuthority::CanonicalBinding
                && in_numeric_batch(implementation.builtin_path)
        })
        .collect::<Vec<_>>();
    let names = owners
        .iter()
        .map(|owner| owner.name)
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), 89, "L05 physical owner scope changed");
    assert_eq!(
        owners.len(),
        names.len(),
        "L05 has one implementation per binding"
    );
    let gpu = builtin_gpu_spec_registrations().collect::<Vec<_>>();
    let fusion = builtin_fusion_spec_registrations().collect::<Vec<_>>();
    // Include both source-owned declarations and declarations naming an owned
    // identity: an incorrect path must not escape the assertion's scope.
    for (kind, name, path) in gpu
        .iter()
        .map(|spec| ("GPU", spec.spec.name, spec.builtin_path))
        .chain(
            fusion
                .iter()
                .map(|spec| ("fusion", spec.spec.name, spec.builtin_path)),
        )
        .filter(|(_, name, path)| names.contains(name) || in_numeric_batch(path))
    {
        assert!(exact_owner(&implementations, name, path).is_some(),
            "{kind} declaration {name} at {path} has no unique exact canonical implementation owner");
    }
    let (mut registered, mut intentional_none, mut absent) = (0, 0, 0);
    for owner in owners {
        let entry = builtin_catalog_primary_entry_by_name(owner.name).expect("canonical entry");
        assert_eq!(
            entry.bindings.len(),
            1,
            "{} requires typed multi-binding coverage",
            owner.name
        );
        let identity = entry.binding_identity(&entry.bindings[0]);
        assert_eq!(owner.binding_variant, Some(identity.variant));
        let gpu_specs = gpu
            .iter()
            .filter(|spec| spec.spec.name == owner.name)
            .collect::<Vec<_>>();
        assert!(
            gpu_specs.len() <= 1,
            "duplicate GPU declarations for {}",
            owner.name
        );
        let fusion_specs = fusion
            .iter()
            .filter(|spec| spec.spec.name == owner.name)
            .collect::<Vec<_>>();
        assert!(
            fusion_specs.len() <= 1,
            "duplicate fusion declarations for {}",
            owner.name
        );
        let resolved = canonical_fusion_declaration(identity);
        if let Some(registration) = fusion_specs.first() {
            let resolved = resolved.unwrap_or_else(|| {
                panic!(
                    "{} registered fusion declaration did not resolve",
                    owner.name
                )
            });
            assert_eq!(resolved.binding, identity);
            assert!(std::ptr::eq(resolved.spec, registration.spec));
            if let Some(semantics) = registration.semantics {
                assert_eq!(semantics.binding, identity);
            }
            registered += 1;
            if registration.spec.elementwise.is_none() && registration.spec.reduction.is_none() {
                intentional_none += 1;
            }
        } else {
            assert!(
                resolved.is_none(),
                "{} resolved an undeclared fusion spec",
                owner.name
            );
            assert_ne!(
                entry.placement.fusion,
                BuiltinFusionPolicy::Candidate,
                "{} advertises fusion candidacy without a declaration",
                owner.name
            );
            absent += 1;
        }
    }
    assert!(
        registered > 0 && intentional_none > 0 && absent > 0,
        "exercise registered templates, declared boundaries, and absent declarations separately"
    );
}

#[test]
fn declaration_owner_join_rejects_matching_name_at_a_child_specification_path() {
    let implementations = builtin_implementation_provenance();
    let owner = implementations
        .iter()
        .find(|owner| {
            owner.authority == BuiltinImplementationAuthority::CanonicalBinding
                && in_numeric_batch(owner.builtin_path)
        })
        .expect("compiled numeric implementation");
    assert!(exact_owner(&implementations, owner.name, owner.builtin_path).is_some());
    let wrong_path = format!("{}::specification", owner.builtin_path);
    assert!(
        exact_owner(&implementations, owner.name, &wrong_path).is_none(),
        "a matching textual name cannot repair mismatched implementation provenance"
    );
}

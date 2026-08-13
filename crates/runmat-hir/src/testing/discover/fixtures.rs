use std::collections::BTreeMap;

use runmat_test::descriptor::{
    FixtureDescriptor, FixtureScope, ProcedureDescriptor, ProcedureKind,
};
use runmat_test::discovery::{
    MaterializationKind, MaterializationLimits, MaterializationRequest, MaterializationResponse,
};
use runmat_test::identity::{FixtureGroupId, FixtureId, ParameterId};

use crate::{HirClass, HirFunction, MethodDeclaration};

use crate::testing::attributes;
use crate::testing::source::source_descriptor;
use crate::testing::{SemanticDiscoveryInput, SemanticTestSource};

pub(super) fn descriptor(
    source: &SemanticTestSource<'_>,
    suite_identity: &str,
    group_id: FixtureGroupId,
    class_name: &str,
    method: &MethodDeclaration,
    function: &HirFunction,
) -> Option<FixtureDescriptor> {
    let (scope, setup) = scope_and_direction(method)?;
    let semantic_path = format!("{class_name}/fixture/{}", method.name.0);
    let procedure = ProcedureDescriptor {
        semantic_path: semantic_path.clone(),
        display_name: method.name.0.clone(),
        kind: if setup {
            ProcedureKind::Fixture
        } else {
            ProcedureKind::Teardown
        },
        source: source_descriptor(source, semantic_path.clone(), function.span),
    };
    Some(FixtureDescriptor {
        id: FixtureId::derive(suite_identity, &semantic_path),
        group_id,
        display_name: method.name.0.clone(),
        scope,
        setup: setup.then_some(procedure.clone()),
        teardown: (!setup).then_some(procedure),
        dependencies: Vec::new(),
    })
}

pub(super) fn is_fixture(method: &MethodDeclaration) -> bool {
    scope_and_direction(method).is_some()
}

pub(super) fn materialization(
    input: &SemanticDiscoveryInput<'_>,
    source: &SemanticTestSource<'_>,
    class: &HirClass,
    class_name: &str,
    suite_identity: &str,
    group_id: &FixtureGroupId,
    materialized: &BTreeMap<&str, &MaterializationResponse>,
) -> (Vec<FixtureDescriptor>, Vec<MaterializationRequest>) {
    let mut fixtures = Vec::new();
    let mut pending = Vec::new();
    for attribute in class
        .declaration
        .declared_attributes
        .iter()
        .filter(|attribute| attribute.name.eq_ignore_ascii_case("SharedTestFixtures"))
    {
        let expression = attribute.value.clone().unwrap_or_default();
        let semantic_path = format!("{class_name}/shared-fixtures");
        let id = ParameterId::derive(&semantic_path, &expression)
            .as_str()
            .to_owned();
        if let Some(response) = materialized.get(id.as_str()) {
            for value in &response.values {
                let fixture_path = format!("{semantic_path}/{}", value.normalized_identity);
                fixtures.push(FixtureDescriptor {
                    id: FixtureId::derive(suite_identity, &fixture_path),
                    group_id: group_id.clone(),
                    display_name: value.name.clone(),
                    scope: FixtureScope::Class,
                    setup: Some(ProcedureDescriptor {
                        semantic_path: fixture_path.clone(),
                        display_name: value.name.clone(),
                        kind: ProcedureKind::Fixture,
                        source: source_descriptor(source, fixture_path, attribute.span),
                    }),
                    teardown: None,
                    dependencies: Vec::new(),
                });
            }
            continue;
        }
        pending.push(MaterializationRequest {
            id,
            program_revision: input.program_revision.clone(),
            kind: MaterializationKind::FixtureExpression,
            semantic_path: semantic_path.clone(),
            source: source_descriptor(source, semantic_path, attribute.span),
            expression,
            limits: MaterializationLimits::default(),
        });
    }
    (fixtures, pending)
}

fn scope_and_direction(method: &MethodDeclaration) -> Option<(FixtureScope, bool)> {
    for (attribute, scope, setup) in [
        ("TestMethodSetup", FixtureScope::Test, true),
        ("TestMethodTeardown", FixtureScope::Test, false),
        ("TestClassSetup", FixtureScope::Class, true),
        ("TestClassTeardown", FixtureScope::Class, false),
    ] {
        if attributes::has(&method.declared_attributes, attribute) {
            return Some((scope, setup));
        }
    }
    None
}

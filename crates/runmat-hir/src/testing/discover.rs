mod classes;
mod fixtures;
mod lineage;
mod scripts;

use std::collections::{BTreeMap, BTreeSet};

use runmat_test::descriptor::{
    ParameterDescriptor, ProcedureDescriptor, ProcedureKind, TestDescriptor, TestRequirements,
};
use runmat_test::discovery::{validate_response, MaterializationResponse, TestDiscovery};
use runmat_test::identity::{FixtureGroupId, SuiteId, TestId, TestIdentityInput};

use crate::{DefPath, DefPathSegment, Span};

use super::source::source_descriptor;
use super::{SemanticDiscoveryInput, SemanticTestSource};

pub fn discover_tests(input: &SemanticDiscoveryInput<'_>) -> TestDiscovery {
    discover(input, &BTreeMap::new())
}

/// Apply bounded metadata responses only after matching them against the
/// initial revision-bound request set, then deterministically rediscover.
pub fn discover_tests_with_materialization(
    input: &SemanticDiscoveryInput<'_>,
    responses: &[MaterializationResponse],
) -> Result<TestDiscovery, Vec<runmat_test::TestDomainError>> {
    let initial = discover_tests(input);
    let requests = initial
        .pending_materialization
        .iter()
        .map(|request| (request.id.as_str(), request))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut errors = Vec::new();
    for response in responses {
        let Some(request) = requests.get(response.request_id.as_str()) else {
            errors.push(runmat_test::TestDomainError::InvalidField {
                field: "materialization.request_id",
                reason: "response was not requested by this discovery revision".into(),
            });
            continue;
        };
        if !seen.insert(response.request_id.as_str()) {
            errors.push(runmat_test::TestDomainError::InvalidField {
                field: "materialization.request_id",
                reason: "duplicate response".into(),
            });
            continue;
        }
        if let Err(error) = validate_response(request, response) {
            errors.push(error);
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let materialized = responses
        .iter()
        .map(|response| (response.request_id.as_str(), response))
        .collect::<BTreeMap<_, _>>();
    let mut discovery = discover(input, &materialized);
    for response in responses {
        discovery.diagnostics.extend(response.diagnostics.clone());
    }
    Ok(discovery)
}

fn discover<'a>(
    input: &SemanticDiscoveryInput<'_>,
    materialized: &BTreeMap<&'a str, &'a MaterializationResponse>,
) -> TestDiscovery {
    let revision_identity = input.program_revision.canonical_identity();
    let mut discovery = TestDiscovery {
        program_revision: input.program_revision.clone(),
        suites: Vec::new(),
        pending_materialization: Vec::new(),
        diagnostics: Vec::new(),
    };
    scripts::discover(input, &revision_identity, &mut discovery);
    classes::discover(input, &revision_identity, materialized, &mut discovery);
    discovery
        .suites
        .sort_by(|left, right| left.id.cmp(&right.id));
    discovery
        .pending_materialization
        .sort_by(|left, right| left.id.cmp(&right.id));
    discovery.diagnostics.sort_by(|left, right| {
        left.code
            .cmp(&right.code)
            .then(left.message.cmp(&right.message))
    });
    discovery
}

#[allow(clippy::too_many_arguments)]
fn descriptor(
    source: &SemanticTestSource<'_>,
    suite_id: SuiteId,
    group_id: FixtureGroupId,
    scheme: &str,
    semantic_path: String,
    display_name: String,
    procedure_kind: ProcedureKind,
    span: Span,
    parameters: Vec<ParameterDescriptor>,
    tags: Vec<String>,
) -> TestDescriptor {
    let parameter_identity = parameters
        .iter()
        .map(|parameter| parameter.id.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let test_id = TestId::derive(&TestIdentityInput {
        owner_identity: source.owner_identity,
        relative_source_identity: source.relative_source_identity,
        semantic_scheme: scheme,
        semantic_item_path: &semantic_path,
        parameter_identity: &parameter_identity,
        fixture_identity: group_id.as_str(),
    });
    TestDescriptor {
        id: test_id,
        suite_id,
        fixture_group_id: group_id,
        display_name: display_name.clone(),
        procedure: ProcedureDescriptor {
            semantic_path: semantic_path.clone(),
            display_name,
            kind: procedure_kind,
            source: source_descriptor(source, semantic_path, span),
        },
        parameters,
        tags,
        requirements: TestRequirements::default(),
    }
}

fn cross_product_parameters(
    left: Vec<Vec<ParameterDescriptor>>,
    right: Vec<Vec<ParameterDescriptor>>,
) -> Vec<Vec<ParameterDescriptor>> {
    let mut combined = Vec::new();
    for left_set in left {
        for right_set in &right {
            let mut values = left_set.clone();
            values.extend(right_set.clone());
            combined.push(values);
        }
    }
    combined
}

fn function_leaf(name: &str) -> &str {
    name.rsplit_once('.').map_or(name, |(_, leaf)| leaf)
}

fn def_path_string(path: &DefPath) -> String {
    let module = path
        .module
        .0
        .iter()
        .map(|segment| segment.0.as_str())
        .collect::<Vec<_>>()
        .join(".");
    let item = path
        .item
        .iter()
        .map(DefPathSegment::display_name)
        .collect::<Vec<_>>()
        .join("/");
    format!("{}::{module}::{item}", path.package.0)
}

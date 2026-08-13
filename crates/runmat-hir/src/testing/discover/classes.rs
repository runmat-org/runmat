use std::collections::BTreeMap;

use runmat_test::descriptor::ProcedureKind;
use runmat_test::discovery::{
    DiscoveredSuite, DiscoveryDiagnostic, DiscoveryDiagnosticSeverity, MaterializationResponse,
    TestDiscovery,
};
use runmat_test::identity::{FixtureGroupId, SuiteId};

use crate::{
    DefPath, DefPathSegment, FunctionId, HirFunction, MethodDeclaration, PackageName,
    QualifiedName, SymbolName,
};

use super::{cross_product_parameters, def_path_string, descriptor};
use super::{fixtures, lineage};
use crate::testing::attributes;
use crate::testing::parameters::{class_parameter_sets, ClassPropertyInput};
use crate::testing::source::source_descriptor;
use crate::testing::{SemanticDiscoveryInput, SemanticTestSource};

pub(super) fn discover(
    input: &SemanticDiscoveryInput<'_>,
    revision_identity: &str,
    materialized: &BTreeMap<&str, &MaterializationResponse>,
    discovery: &mut TestDiscovery,
) {
    let records = lineage::class_records(input.sources);
    let by_name = lineage::by_name(&records);

    for (record_index, record) in records.iter().enumerate() {
        let lineage =
            match lineage::test_case_lineage(record_index, &records, &by_name, input.sources) {
                lineage::Lineage::TestCase(lineage) => lineage,
                lineage::Lineage::NotTestCase => continue,
                lineage::Lineage::Invalid { code, message } => {
                    let source = &input.sources[record.source_index];
                    let class = &source.assembly.classes[record.class_index];
                    discovery.diagnostics.push(DiscoveryDiagnostic {
                        code: code.into(),
                        message,
                        severity: DiscoveryDiagnosticSeverity::Error,
                        source: Some(source_descriptor(
                            source,
                            record.name.clone(),
                            class.declaration.span,
                        )),
                    });
                    continue;
                }
            };
        let source = &input.sources[record.source_index];
        let class = &source.assembly.classes[record.class_index];
        let semantic_suite = format!(
            "{}/{}#class:{}",
            source.owner_identity,
            source.relative_source_identity.replace('\\', "/"),
            record.name
        );
        let suite_id = SuiteId::derive(revision_identity, &semantic_suite);
        let group_id = FixtureGroupId::derive(suite_id.as_str(), &record.name);
        let mut class_tags = Vec::new();
        let mut methods =
            BTreeMap::<String, (&SemanticTestSource<'_>, &MethodDeclaration, &HirFunction)>::new();
        let mut fixture_methods =
            BTreeMap::<String, (&SemanticTestSource<'_>, &MethodDeclaration, &HirFunction)>::new();
        let mut properties =
            BTreeMap::<String, (&SemanticTestSource<'_>, ClassPropertyInput)>::new();
        let mut materialized_fixtures = Vec::new();
        let pending_before = discovery.pending_materialization.len();
        for ancestor_index in lineage {
            let ancestor_record = &records[ancestor_index];
            let ancestor_source = &input.sources[ancestor_record.source_index];
            let ancestor = &ancestor_source.assembly.classes[ancestor_record.class_index];
            class_tags.extend(attributes::tags(&ancestor.declaration.declared_attributes));
            for property in &ancestor.declaration.properties {
                properties.insert(
                    property.name.0.to_ascii_lowercase(),
                    (
                        ancestor_source,
                        ClassPropertyInput {
                            declaration: property.clone(),
                            default: ancestor.property_default(&property.name).cloned(),
                        },
                    ),
                );
            }
            let (ancestor_fixtures, pending) = fixtures::materialization(
                input,
                ancestor_source,
                ancestor,
                &record.name,
                suite_id.as_str(),
                &group_id,
                materialized,
            );
            materialized_fixtures.extend(ancestor_fixtures);
            discovery.pending_materialization.extend(pending);
            for method in &ancestor.declaration.methods {
                let Some(function) = function_by_id(ancestor_source, method.function) else {
                    continue;
                };
                methods.insert(
                    method.name.0.to_ascii_lowercase(),
                    (ancestor_source, method, function),
                );
                if fixtures::is_fixture(method) {
                    fixture_methods.insert(
                        method.name.0.to_ascii_lowercase(),
                        (ancestor_source, method, function),
                    );
                }
            }
        }
        class_tags.sort();
        class_tags.dedup();

        let mut parameter_sets = vec![Vec::new()];
        for (parameter_source, properties) in grouped_properties(properties.into_values()) {
            let (source_sets, pending) = class_parameter_sets(
                parameter_source,
                &record.name,
                &properties,
                &input.program_revision,
                materialized,
            );
            parameter_sets = cross_product_parameters(parameter_sets, source_sets);
            discovery.pending_materialization.extend(pending);
        }

        let mut fixtures = fixture_methods
            .into_values()
            .filter_map(|(method_source, method, function)| {
                fixtures::descriptor(
                    method_source,
                    suite_id.as_str(),
                    group_id.clone(),
                    &record.name,
                    method,
                    function,
                )
            })
            .collect::<Vec<_>>();
        fixtures.extend(materialized_fixtures);
        fixtures.sort_by(|left, right| left.id.cmp(&right.id));
        let mut tests = Vec::new();
        for (_, (method_source, method, function)) in methods {
            if !attributes::has(&method.declared_attributes, "Test") {
                continue;
            }
            let mut tags = class_tags.clone();
            tags.extend(attributes::tags(&method.declared_attributes));
            tags.sort();
            tags.dedup();
            for parameters in &parameter_sets {
                let parameter_suffix = if parameters.is_empty() {
                    String::new()
                } else {
                    format!(
                        "[{}]",
                        parameters
                            .iter()
                            .map(|parameter| {
                                format!("{}={}", parameter.name, parameter.normalized_identity)
                            })
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                };
                let path = DefPath {
                    package: PackageName(method_source.owner_identity.to_owned()),
                    module: QualifiedName(vec![SymbolName(record.name.clone())]),
                    item: vec![
                        DefPathSegment::Class(SymbolName(record.name.clone())),
                        DefPathSegment::Method(SymbolName(method.name.0.clone())),
                    ],
                };
                tests.push(descriptor(
                    method_source,
                    suite_id.clone(),
                    group_id.clone(),
                    "class-method",
                    def_path_string(&path),
                    format!("{}/{}{}", record.name, method.name.0, parameter_suffix),
                    ProcedureKind::Method,
                    function.span,
                    parameters.clone(),
                    tags.clone(),
                ));
            }
        }
        tests.sort_by(|left, right| left.id.cmp(&right.id));
        let has_pending = discovery.pending_materialization.len() > pending_before;
        if tests.is_empty() && !has_pending {
            continue;
        }
        discovery.suites.push(DiscoveredSuite {
            id: suite_id,
            fixture_group_id: group_id,
            display_name: record.name.clone(),
            source: source_descriptor(source, record.name.clone(), class.declaration.span),
            fixtures,
            tests,
        });
    }
}

fn grouped_properties<'a>(
    properties: impl Iterator<Item = (&'a SemanticTestSource<'a>, ClassPropertyInput)>,
) -> Vec<(&'a SemanticTestSource<'a>, Vec<ClassPropertyInput>)> {
    let mut grouped: Vec<(&SemanticTestSource<'_>, Vec<ClassPropertyInput>)> = Vec::new();
    for (source, property) in properties {
        if let Some((_, values)) = grouped
            .iter_mut()
            .find(|(candidate, _)| std::ptr::eq(*candidate, source))
        {
            values.push(property);
        } else {
            grouped.push((source, vec![property]));
        }
    }
    grouped
}

fn function_by_id<'a>(
    source: &'a SemanticTestSource<'_>,
    id: FunctionId,
) -> Option<&'a HirFunction> {
    source
        .assembly
        .functions
        .iter()
        .find(|function| function.id == id)
}

use super::*;
use crate::{BytecodeSourceRef, Instr, PrivateSourceAlias, PrivateSourceDirectory};
use runmat_execution::{Digest, FunctionBindingOrigin};
use runmat_types::{FunctionSourceRole, ProgramFunctionId, ProgramSourceId};

struct UnitFixture {
    unit: RuntimeBoundExecutableUnit,
    _source: runmat_runtime::source_context::RuntimeSourceLease,
}

fn origin(program: &'static [u8], function: usize) -> FunctionBindingOrigin {
    FunctionBindingOrigin {
        program: Digest::sha256(program),
        function: ProgramFunctionId(function as u32),
    }
}

fn runtime() -> runmat_runtime::context::RuntimeContext {
    runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
}

fn unit(
    runtime: &runmat_runtime::context::RuntimeContext,
    program: &'static [u8],
    names: [&str; 2],
) -> UnitFixture {
    let source = runmat_runtime::source_context::lease_source_context_in_context(
        runtime,
        format!("{}.m", names[0]),
        None,
        "fixture",
    )
    .expect("source");
    let source_ref = Some(BytecodeSourceRef::Runtime(source.source_ref()));
    let helper = FunctionBytecode {
        function: FunctionId(2),
        display_name: names[1].into(),
        source_id: source_ref,
        source_file: Some(format!("{}.m", names[0])),
        source_role: FunctionSourceRole::Local,
        portable_function: ProgramFunctionId(2),
        binding_origin: Some(origin(program, 2)),
        ..Default::default()
    };
    let primary = FunctionBytecode {
        function: FunctionId(1),
        display_name: names[0].into(),
        source_id: source_ref,
        source_file: Some(format!("{}.m", names[0])),
        source_role: FunctionSourceRole::Primary,
        portable_function: ProgramFunctionId(1),
        binding_origin: Some(origin(program, 1)),
        instructions: vec![Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)],
        ..Default::default()
    };
    let registry = FunctionRegistry::new(HashMap::from([
        (FunctionId(1), primary),
        (FunctionId(2), helper),
    ]));
    UnitFixture {
        unit: RuntimeBoundExecutableUnit::for_test(registry, runtime.clone()),
        _source: source,
    }
}

fn limits() -> RetainedCallableStageLimits {
    RetainedCallableStageLimits {
        max_units: 8,
        max_functions: 16,
        max_instructions: 32,
    }
}

#[test]
fn retained_publication_requires_a_canonical_admitted_source_product() {
    let runtime = runtime();
    let fixture = unit(&runtime, b"test-only-source", ["primary", "helper"]);
    let staged = prepare_retained_callable_units(
        &runtime,
        &FunctionRegistry::default(),
        vec![fixture.unit],
        limits(),
    )
    .expect("test-only body may be inspected during staging");
    assert!(matches!(
        staged.into_parts(),
        Err(RetainedCallableStageError::RetainedSourceOwner(
            "staged imported unit has no canonical V3 product"
        ))
    ));
}

#[test]
fn overlapping_originating_ids_are_fresh_and_retained_only() {
    let runtime = runtime();
    let destination = FunctionRegistry::new(HashMap::from([(
        FunctionId(9),
        FunctionBytecode {
            function: FunctionId(9),
            display_name: "published".into(),
            ..Default::default()
        },
    )]));
    let first = unit(&runtime, b"first-program", ["first", "first_helper"]);
    let second = unit(&runtime, b"second-program", ["second", "second_helper"]);
    let prepared = prepare_retained_callable_units(
        &runtime,
        &destination,
        vec![first.unit, second.unit],
        limits(),
    )
    .expect("retained graph");

    assert_eq!(
        prepared.registry().resolve_name("published"),
        Some(FunctionId(9))
    );
    assert!(prepared.registry().resolve_name("first").is_none());
    assert!(prepared.registry().resolve_name("second").is_none());
    let first_info = prepared
        .exact_origin_facts()
        .iter()
        .find(|info| info.binding_origin == Some(origin(b"first-program", 1)))
        .expect("first origin");
    let second_info = prepared
        .exact_origin_facts()
        .iter()
        .find(|info| info.binding_origin == Some(origin(b"second-program", 1)))
        .expect("second origin");
    assert_ne!(first_info.function, 1);
    assert_ne!(second_info.function, 1);
    assert_ne!(first_info.function, second_info.function);

    let graph = first_info.definition_owners.as_ref().expect("owner graph");
    let (root_id, root_owner) = graph
        .iter()
        .find(|(function, _)| *function == first_info.function)
        .expect("root owner");
    let root_lease = FunctionDefinitionOwner::issue(root_owner, *root_id);
    let body = prepared
        .registry()
        .get_owned(FunctionId(*root_id), &root_lease)
        .expect("retained body");
    let Instr::CallSemanticFunctionMulti(helper_id, ..) = &body.instructions[0] else {
        panic!("static helper call")
    };
    assert_ne!(*helper_id, FunctionId(2));
    assert!(graph.iter().any(|(function, _)| *function == helper_id.0));
    assert!(prepared
        .registry()
        .functions_for_source_ref(body.source_id.expect("source"))
        .is_empty());
}

#[test]
fn retained_private_companion_is_leased_without_a_static_call_edge() {
    let runtime = runtime();
    let caller_source = runmat_runtime::source_context::lease_source_context_in_context(
        &runtime,
        "entry.m",
        None,
        "function p = entry; p = which('helper'); end",
    )
    .expect("caller source");
    let helper_source = runmat_runtime::source_context::lease_source_context_in_context(
        &runtime,
        "private/helper.m",
        None,
        "function y = helper; y = 7; end",
    )
    .expect("private source");
    let alias = PrivateSourceAlias {
        owner_identity: "fixture-program".into(),
        relative_directory: "source".into(),
    };
    let owner = PrivateSourceDirectory::captured(None, vec![alias]);
    let caller = FunctionBytecode {
        function: FunctionId(1),
        display_name: "entry".into(),
        source_id: Some(BytecodeSourceRef::Runtime(caller_source.source_ref())),
        source_file: Some("source/entry.m".into()),
        private_source_owner: Some(owner.clone()),
        source_role: FunctionSourceRole::Primary,
        portable_function: ProgramFunctionId(1),
        binding_origin: Some(origin(b"private-only", 1)),
        ..Default::default()
    };
    let helper = FunctionBytecode {
        function: FunctionId(2),
        display_name: "__private__.helper".into(),
        source_id: Some(BytecodeSourceRef::Runtime(helper_source.source_ref())),
        source_file: Some("source/private/helper.m".into()),
        private_source_owner: Some(owner.clone()),
        source_role: FunctionSourceRole::Primary,
        portable_function: ProgramFunctionId(2),
        binding_origin: Some(origin(b"private-only", 2)),
        ..Default::default()
    };
    let mut bodies = HashMap::from([(FunctionId(1), caller), (FunctionId(2), helper)]);
    let mut _sibling_leases = Vec::new();
    for id in 3..=10 {
        let name = format!("helper{id}");
        let file = format!("source/private/{name}.m");
        let lease = runmat_runtime::source_context::lease_source_context_in_context(
            &runtime,
            file.as_str(),
            None,
            "fixture",
        )
        .expect("private sibling source");
        bodies.insert(
            FunctionId(id),
            FunctionBytecode {
                function: FunctionId(id),
                display_name: format!("__private__.{name}"),
                source_id: Some(BytecodeSourceRef::Runtime(lease.source_ref())),
                source_file: Some(file),
                private_source_owner: Some(owner.clone()),
                source_role: FunctionSourceRole::Primary,
                portable_function: ProgramFunctionId(id as u32),
                binding_origin: Some(origin(b"private-only", id)),
                ..Default::default()
            },
        );
        _sibling_leases.push(lease);
    }
    let registry = FunctionRegistry::new(bodies);
    let unit = RuntimeBoundExecutableUnit::for_test(registry, runtime.clone());
    let prepared = prepare_retained_callable_units(
        &runtime,
        &FunctionRegistry::default(),
        vec![unit],
        limits(),
    )
    .expect("retained private graph");
    let caller_fact = prepared
        .exact_origin_facts()
        .iter()
        .find(|fact| fact.binding_origin == Some(origin(b"private-only", 1)))
        .expect("caller origin");
    let helper_fact = prepared
        .exact_origin_facts()
        .iter()
        .find(|fact| fact.binding_origin == Some(origin(b"private-only", 2)))
        .expect("helper origin");
    let graph = caller_fact
        .definition_owners
        .as_ref()
        .expect("source graph");
    assert!(graph.iter().any(|(id, _)| *id == helper_fact.function));
    assert!(std::sync::Arc::ptr_eq(
        graph,
        helper_fact
            .definition_owners
            .as_ref()
            .expect("shared partition graph")
    ));
    assert_eq!(graph.len(), 10);
    for sibling in prepared.exact_origin_facts().iter() {
        assert!(std::sync::Arc::ptr_eq(
            graph,
            sibling
                .definition_owners
                .as_ref()
                .expect("shared sibling graph")
        ));
    }
    assert!(prepared.registry().resolve_name("helper").is_none());
}

#[test]
fn retained_static_edge_expands_observation_only_private_scope_in_another_partition() {
    let runtime = runtime();
    let mut _leases = Vec::new();
    let mut bodies = HashMap::new();
    for (id, name, file, owner_name) in [
        (1, "entry", "a/entry.m", "a"),
        (2, "other", "b/other.m", "b"),
        (3, "__private__.helper", "b/private/helper.m", "b"),
    ] {
        let lease = runmat_runtime::source_context::lease_source_context_in_context(
            &runtime, file, None, "fixture",
        )
        .expect("source");
        let owner = PrivateSourceDirectory::captured(
            None,
            vec![PrivateSourceAlias {
                owner_identity: "transitive-program".into(),
                relative_directory: owner_name.into(),
            }],
        );
        bodies.insert(
            FunctionId(id),
            FunctionBytecode {
                function: FunctionId(id),
                display_name: name.into(),
                source_id: Some(BytecodeSourceRef::Runtime(lease.source_ref())),
                source_file: Some(file.into()),
                private_source_owner: Some(owner),
                source_role: FunctionSourceRole::Primary,
                portable_function: ProgramFunctionId(id as u32),
                binding_origin: Some(origin(b"transitive-program", id)),
                instructions: if id == 1 {
                    vec![Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)]
                } else {
                    Vec::new()
                },
                ..Default::default()
            },
        );
        _leases.push(lease);
    }
    let unit = RuntimeBoundExecutableUnit::for_test(FunctionRegistry::new(bodies), runtime.clone());
    let prepared = prepare_retained_callable_units(
        &runtime,
        &FunctionRegistry::default(),
        vec![unit],
        limits(),
    )
    .expect("cross-partition retained graph");
    let entry = prepared
        .exact_origin_facts()
        .iter()
        .find(|fact| fact.binding_origin == Some(origin(b"transitive-program", 1)))
        .expect("entry");
    let private = prepared
        .exact_origin_facts()
        .iter()
        .find(|fact| fact.binding_origin == Some(origin(b"transitive-program", 3)))
        .expect("private companion");
    assert!(entry
        .definition_owners
        .as_ref()
        .unwrap()
        .iter()
        .any(|(id, _)| *id == private.function));
    assert!(prepared.registry().resolve_name("helper").is_none());
}

#[test]
fn incomplete_graph_fails_without_changing_destination() {
    let runtime = runtime();
    let fixture = unit(&runtime, b"incomplete", ["primary", "helper"]);
    let mut incomplete = fixture.unit.registry().clone();
    incomplete.functions.remove(&FunctionId(2));
    let unit = RuntimeBoundExecutableUnit::for_test(incomplete, fixture.unit.runtime().clone());
    let destination = FunctionRegistry::default();
    assert!(matches!(
        prepare_retained_callable_units(&runtime, &destination, vec![unit], limits()),
        Err(RetainedCallableStageError::IncompleteDefinitionGraph)
    ));
    assert!(destination.functions.is_empty());
    assert!(destination.names.is_empty());
}

#[test]
fn unit_bound_to_foreign_session_is_rejected_before_destination_mutation() {
    let destination_runtime = runtime();
    let foreign_runtime = runtime();
    let foreign = unit(&foreign_runtime, b"foreign-session", ["primary", "helper"]);
    let destination = FunctionRegistry::new(HashMap::from([(
        FunctionId(9),
        FunctionBytecode {
            function: FunctionId(9),
            display_name: "published".into(),
            ..Default::default()
        },
    )]));

    assert!(matches!(
        prepare_retained_callable_units(
            &destination_runtime,
            &destination,
            vec![foreign.unit],
            limits(),
        ),
        Err(RetainedCallableStageError::ForeignSourceOwner)
    ));
    assert_eq!(destination.resolve_name("published"), Some(FunctionId(9)));
    assert_eq!(destination.functions.len(), 1);
}

#[test]
fn conflicting_origin_is_rejected_in_both_input_orders() {
    for reverse in [false, true] {
        let runtime = runtime();
        let first = unit(&runtime, b"same-program", ["primary", "helper"]);
        let second = unit(&runtime, b"same-program", ["primary", "helper"]);
        let mut conflicting_registry = second.unit.registry().clone();
        conflicting_registry
            .functions
            .get_mut(&FunctionId(2))
            .expect("helper")
            .instructions
            .push(Instr::LoadInt(runmat_value::IntValue::I64(91)));
        let conflicting = RuntimeBoundExecutableUnit::for_test(
            conflicting_registry,
            second.unit.runtime().clone(),
        );
        let units = if reverse {
            vec![conflicting, first.unit]
        } else {
            vec![first.unit, conflicting]
        };
        assert!(matches!(
            prepare_retained_callable_units(
                &runtime,
                &FunctionRegistry::default(),
                units,
                limits()
            ),
            Err(RetainedCallableStageError::ConflictingOrigin)
        ));
    }
}

#[test]
fn same_origin_and_body_with_different_local_packaging_is_deduplicated() {
    let runtime = runtime();
    let first = unit(&runtime, b"same-envelope", ["primary", "helper"]);
    let second = unit(&runtime, b"same-envelope", ["primary", "helper"]);
    let local_remap = HashMap::from([
        (FunctionId(1), FunctionId(101)),
        (FunctionId(2), FunctionId(102)),
    ]);
    let differently_packaged = FunctionRegistry::new(
        second
            .unit
            .registry()
            .functions
            .values()
            .cloned()
            .map(|mut body| {
                function_remap::remap_closed_function(&mut body, &local_remap)
                    .expect("closed local projection");
                (body.function, body)
            })
            .collect(),
    );
    let differently_packaged =
        RuntimeBoundExecutableUnit::for_test(differently_packaged, second.unit.runtime().clone());
    let prepared = prepare_retained_callable_units(
        &runtime,
        &FunctionRegistry::default(),
        vec![first.unit, differently_packaged],
        limits(),
    )
    .expect("identical canonical unit");
    assert_eq!(prepared.exact_origin_facts().len(), 2);
    let mut functions = prepared
        .exact_origin_facts()
        .iter()
        .map(|info| info.function)
        .collect::<Vec<_>>();
    functions.sort_unstable();
    functions.dedup();
    assert_eq!(functions.len(), 2);
}

#[test]
fn fingerprints_allow_reused_portable_ordinals_across_exact_origins() {
    let dependency = FunctionBytecode {
        function: FunctionId(7),
        display_name: "dependency".into(),
        portable_function: ProgramFunctionId(0),
        binding_origin: Some(origin(b"dependency-program", 0)),
        ..Default::default()
    };
    let root = FunctionBytecode {
        function: FunctionId(11),
        display_name: "root".into(),
        portable_function: ProgramFunctionId(0),
        binding_origin: Some(origin(b"root-program", 0)),
        instructions: vec![Instr::CallSemanticFunctionMulti(FunctionId(7), 0, 1)],
        ..Default::default()
    };
    let base = FunctionRegistry::new(HashMap::from([
        (root.function, root.clone()),
        (dependency.function, dependency.clone()),
    ]));
    let base_fingerprints = fingerprints::normalized(&base).expect("multi-origin registry");

    let unrelated = FunctionBytecode {
        function: FunctionId(3),
        display_name: "unrelated".into(),
        portable_function: ProgramFunctionId(0),
        binding_origin: Some(origin(b"another-program", 0)),
        ..Default::default()
    };
    let wider = FunctionRegistry::new(HashMap::from([
        (root.function, root),
        (dependency.function, dependency),
        (unrelated.function, unrelated),
    ]));
    let wider_fingerprints = fingerprints::normalized(&wider).expect("wider registry");
    assert_eq!(
        base_fingerprints[&origin(b"root-program", 0)],
        wider_fingerprints[&origin(b"root-program", 0)],
        "unrelated packaging must not alter a retained definition fingerprint"
    );
}

fn portable_source_map(
    entries: impl IntoIterator<Item = (&'static str, &'static str, &'static str)>,
) -> runmat_execution::ExecutableSourceMapV2 {
    runmat_execution::ExecutableSourceMapV2::new(
        entries
            .into_iter()
            .enumerate()
            .map(|(ordinal, (owner_identity, relative_path, text))| {
                runmat_execution::ExecutableSourceRecordV2 {
                    source_id: ProgramSourceId(u32::try_from(ordinal).expect("source ordinal")),
                    owner_identity: owner_identity.into(),
                    relative_path: relative_path.into(),
                    content_revision: Digest::sha256(text.as_bytes()),
                    display_name: relative_path.into(),
                    text: text.into(),
                }
            })
            .collect(),
    )
    .expect("valid source map")
}

#[test]
fn admitted_fingerprint_resolves_portable_source_position_to_exact_identity() {
    let exact_origin = origin(b"retained-source-program", 4);
    let body = |source| FunctionBytecode {
        function: FunctionId(4),
        display_name: "retainedSource".into(),
        source_id: Some(BytecodeSourceRef::Program(source)),
        source_file: Some("retained.m".into()),
        portable_function: ProgramFunctionId(4),
        binding_origin: Some(exact_origin),
        instructions: vec![Instr::LoadInt(runmat_value::IntValue::I64(7))],
        ..Default::default()
    };
    let primary_registry =
        FunctionRegistry::new(HashMap::from([(FunctionId(4), body(ProgramSourceId(0)))]));
    let companion_registry =
        FunctionRegistry::new(HashMap::from([(FunctionId(4), body(ProgramSourceId(1)))]));
    let primary = portable_source_map([("session", "retained.m", "retained source")]);
    let companion = portable_source_map([
        ("request", "unrelated.m", "unrelated source"),
        ("session", "retained.m", "retained source"),
    ]);

    let primary = fingerprints::normalized_admitted(&primary_registry, &primary)
        .expect("primary fingerprint");
    let companion = fingerprints::normalized_admitted(&companion_registry, &companion)
        .expect("companion fingerprint");
    assert_eq!(primary[&exact_origin], companion[&exact_origin]);
}

#[test]
fn admitted_fingerprint_binds_portable_source_owner_path_and_content() {
    let exact_origin = origin(b"retained-source-program", 4);
    let body = FunctionBytecode {
        function: FunctionId(4),
        display_name: "retainedSource".into(),
        source_id: Some(BytecodeSourceRef::Program(ProgramSourceId(0))),
        source_file: Some("retained.m".into()),
        portable_function: ProgramFunctionId(4),
        binding_origin: Some(exact_origin),
        instructions: vec![Instr::LoadInt(runmat_value::IntValue::I64(7))],
        ..Default::default()
    };
    let registry = FunctionRegistry::new(HashMap::from([(FunctionId(4), body)]));
    let fingerprint = |owner, path, text| {
        let map = portable_source_map([(owner, path, text)]);
        fingerprints::normalized_admitted(&registry, &map).expect("admitted fingerprint")
            [&exact_origin]
            .clone()
    };
    let expected = fingerprint("session", "retained.m", "retained source");
    assert_ne!(
        expected,
        fingerprint("foreign", "retained.m", "retained source")
    );
    assert_ne!(
        expected,
        fingerprint("session", "moved.m", "retained source")
    );
    assert_ne!(
        expected,
        fingerprint("session", "retained.m", "changed source")
    );

    let missing = FunctionRegistry::new(HashMap::from([(
        FunctionId(4),
        FunctionBytecode {
            source_id: Some(BytecodeSourceRef::Program(ProgramSourceId(1))),
            ..registry.functions[&FunctionId(4)].clone()
        },
    )]));
    let map = portable_source_map([("session", "retained.m", "retained source")]);
    assert!(matches!(
        fingerprints::normalized_admitted(&missing, &map),
        Err(RetainedCallableStageError::MissingPortableSource)
    ));
}

#[test]
fn fingerprint_binds_the_exact_dependency_origins() {
    let runtime = runtime();
    let first = unit(&runtime, b"root-program", ["root", "dependency"]);
    let mut retargeted = first.unit.registry().clone();
    retargeted
        .functions
        .get_mut(&FunctionId(2))
        .expect("dependency")
        .binding_origin = Some(origin(b"replacement-dependency", 2));
    let retargeted = RuntimeBoundExecutableUnit::for_test(retargeted, runtime.clone());
    assert!(matches!(
        prepare_retained_callable_units(
            &runtime,
            &FunctionRegistry::default(),
            vec![first.unit, retargeted],
            limits(),
        ),
        Err(RetainedCallableStageError::ConflictingOrigin)
    ));
}

#[test]
fn complete_work_census_precedes_definition_decoding_and_cloning() {
    let runtime = runtime();
    let fixture = unit(&runtime, b"over-limit", ["primary", "helper"]);
    let mut invalid = fixture.unit.registry().clone();
    invalid
        .functions
        .get_mut(&FunctionId(1))
        .unwrap()
        .binding_origin = None;
    let unit = RuntimeBoundExecutableUnit::for_test(invalid, fixture.unit.runtime().clone());
    let error = prepare_retained_callable_units(
        &runtime,
        &FunctionRegistry::default(),
        vec![unit],
        RetainedCallableStageLimits {
            max_units: 1,
            max_functions: 1,
            max_instructions: usize::MAX,
        },
    )
    .expect_err("function census must reject before inspecting origins");
    assert!(matches!(
        error,
        RetainedCallableStageError::Limit {
            field: "callable_artifact.functions",
            limit: 1
        }
    ));
}

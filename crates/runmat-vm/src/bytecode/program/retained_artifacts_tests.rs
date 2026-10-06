use super::*;
use crate::{BytecodeSourceRef, Instr};
use runmat_execution::{Digest, FunctionBindingOrigin};
use runmat_types::{FunctionSourceRole, ProgramFunctionId};

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
                function_remap::remap_closed_function(&mut body, &local_remap);
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

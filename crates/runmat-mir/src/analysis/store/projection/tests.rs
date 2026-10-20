use super::*;
use crate::analysis::analyze_assembly;
use runmat_types::{CallableIdentity, ProgramSourceId, ValueKindFact};

fn assembly() -> crate::MirAssembly {
    let ast = runmat_parser::parse(
        "answer = top(4); selected = target();\nfunction y = top(x); y = leaf(x); end\nfunction y = leaf(x); if x < 0; error('negative'); end; y = x + 1; end\nfunction h = target(); h = @leaf; end",
    ).unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mut mir = crate::lowering::lower_assembly(&hir.assembly).unwrap();
    // A companion source participates in the same assembled analysis before
    // publication; changing coordinates must not lose its callee facts.
    for metadata in mir.functions.values_mut() {
        if metadata.name.0 == "leaf" {
            metadata.source = ProgramSourceId(1);
        }
    }
    mir
}

#[test]
fn local_publication_preserves_cross_source_flow_effects_and_callable_references() {
    let mut mir = assembly();
    let before = analyze_assembly(&mir);
    let named = |name: &str| {
        *mir.functions
            .iter()
            .find(|(_, metadata)| metadata.name.0 == name)
            .unwrap()
            .0
    };
    let leaf = named("leaf");
    let top = named("top");
    let target = named("target");
    assert_eq!(
        before.function_for_local(top).unwrap().outputs,
        before.function_for_local(leaf).unwrap().outputs
    );
    assert_eq!(
        before.function_for_local(top).unwrap().effects,
        before.function_for_local(leaf).unwrap().effects
    );
    assert!(before
        .function_for_local(leaf)
        .unwrap()
        .effects
        .0
        .contains(&runmat_types::EffectKind::MayThrow));
    let ValueKindFact::Callable(callable) =
        &before.function_for_local(target).unwrap().outputs[0].kind
    else {
        panic!("retained callable output");
    };
    assert_eq!(
        callable.identity,
        Some(CallableIdentity::BoundFunction(leaf))
    );
    let rename = mir
        .bodies
        .keys()
        .map(|id| (*id, FunctionId(id.0 + 100)))
        .collect::<HashMap<_, _>>();
    let owners = before
        .functions
        .iter()
        .map(|facts| {
            let Some(CallableIdentity::BoundFunction(local)) = &facts.callable.identity else {
                panic!("self binding");
            };
            (rename[local], facts.function)
        })
        .collect();
    let retained = before.rebind_eval_local_functions(&rename).unwrap();
    crate::remap_function_ids(&mut mir, &rename).unwrap();
    let independently_analyzed = analyze_assembly(&mir)
        .project_product_functions(&owners)
        .unwrap();
    assert_eq!(retained, independently_analyzed);
    let ValueKindFact::Callable(callable) = &retained
        .function_for_local(rename[&target])
        .unwrap()
        .outputs[0]
        .kind
    else {
        panic!("retained callable output");
    };
    assert_eq!(
        callable.identity,
        Some(CallableIdentity::BoundFunction(rename[&leaf]))
    );
    assert!(retained.function_for_local(target).is_none());
}

#[test]
fn product_projection_rejects_colliding_and_unresolved_self_bindings() {
    let facts = analyze_assembly(&assembly());
    let locals = facts
        .functions
        .iter()
        .map(|facts| {
            let Some(CallableIdentity::BoundFunction(local)) = &facts.callable.identity else {
                panic!("self binding");
            };
            *local
        })
        .collect::<Vec<_>>();
    assert!(facts
        .project_product_functions(&BTreeMap::from([
            (locals[0], ProgramFunctionId(9)),
            (locals[1], ProgramFunctionId(9))
        ]))
        .is_err());
    assert!(facts
        .project_product_functions(&BTreeMap::from([(
            FunctionId(usize::MAX),
            ProgramFunctionId(9)
        )]))
        .is_err());
    assert!(facts
        .rebind_eval_local_functions(&HashMap::from([
            (locals[0], FunctionId(99)),
            (locals[1], FunctionId(99))
        ]))
        .is_err());
}

#[test]
fn nested_parallel_and_future_facts_keep_their_producer_owner() {
    use runmat_types::{
        CompositeFact, DistributedFact, DistributedOwner, DistributedValueId, ExecutionFact,
        FutureStateFact, LabCount, ParallelRegionId, RegionId, ValueFact, ValueSequenceFact,
    };
    let mut facts = analyze_assembly(&assembly());
    let producer = facts.functions[1].function;
    let region = ParallelRegionId(RegionId {
        function: producer,
        ordinal: 3,
    });
    let nested = ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Future {
        state: FutureStateFact::Lazy,
        output: ValueSequenceFact::fixed(vec![ValueFact::scalar(ValueKindFact::Composite(
            CompositeFact {
                owner: region,
                labs: LabCount(2),
                value: Box::new(ValueFact::scalar(ValueKindFact::Distributed(
                    DistributedFact {
                        id: DistributedValueId {
                            function: producer,
                            ordinal: 4,
                        },
                        owner: DistributedOwner::Region(region),
                        scheme: None,
                        value: Box::new(ValueFact::scalar(ValueKindFact::Logical)),
                        materializable: true,
                    },
                ))),
            },
        ))]),
    }));
    facts.functions[0].outputs = vec![nested.clone()];
    facts.functions[0].callable.outputs = vec![nested];
    let owners = facts
        .functions
        .iter()
        .map(|facts| {
            let Some(CallableIdentity::BoundFunction(local)) = &facts.callable.identity else {
                panic!("self binding");
            };
            (*local, ProgramFunctionId(facts.function.0 + 10))
        })
        .collect::<BTreeMap<_, _>>();
    let projected = facts.project_product_functions(&owners).unwrap();
    let ValueKindFact::Execution(ExecutionFact::Future { output, .. }) =
        &projected.functions[0].outputs[0].kind
    else {
        panic!("future output");
    };
    let ValueKindFact::Composite(composite) = &output.outputs[0].kind else {
        panic!("composite output");
    };
    let ValueKindFact::Distributed(distributed) = &composite.value.kind else {
        panic!("distributed output");
    };
    assert_eq!(
        composite.owner.0.function,
        ProgramFunctionId(producer.0 + 10)
    );
    assert_eq!(distributed.id.function, composite.owner.0.function);
    assert_eq!(distributed.owner, DistributedOwner::Region(composite.owner));
    // A bare ordinal cannot certify facts from an excluded revision. Normal
    // retained calls have no callee body/summary in the assembled frontend;
    // their original dependency product remains the authority, not this map.
    let mut incomplete = owners;
    incomplete.retain(|_, portable| *portable != ProgramFunctionId(producer.0 + 10));
    assert!(facts.project_product_functions(&incomplete).is_err());
}

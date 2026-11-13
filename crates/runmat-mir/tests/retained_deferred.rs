//! Retained declarations occupy a different namespace from caller-local ordinals.
use std::collections::HashMap;

use runmat_hir::{CallableIdentity, FunctionId, FunctionModifiers, LoweringContext};
use runmat_mir::{MirCallee, MirRvalue, MirStmtKind};

#[test]
fn original_async_declaration_survives_wire_and_local_function_remapping() {
    let ast = runmat_parser::parse(
        "f = remote(5); g = local(2);\nfunction y = local(x)\ny = x + 1;\nend",
    )
    .unwrap();
    let bound = HashMap::from([("remote".to_owned(), FunctionId(0))]);
    let modifiers = HashMap::from([(
        FunctionId(0),
        FunctionModifiers {
            is_async: true,
            isolated: false,
        },
    )]);
    let context = LoweringContext::empty()
        .with_bound_functions(&bound)
        .with_function_modifiers(&modifiers);
    let hir = runmat_hir::lower(&ast, &context).unwrap().assembly;
    let cold: runmat_hir::HirAssembly =
        serde_json::from_slice(&serde_json::to_vec(&hir).unwrap()).unwrap();
    assert_eq!(
        cold.bound_function_modifiers[&FunctionId(0)],
        modifiers[&FunctionId(0)]
    );
    let mut mir = runmat_mir::lowering::lower_assembly(&cold).unwrap();
    assert!(!mir.functions[&FunctionId(0)].modifiers.is_async);
    check_calls(&mir, FunctionId(0));
    runmat_mir::remap_function_ids(&mut mir, &HashMap::from([(FunctionId(0), FunctionId(900))]))
        .unwrap();
    assert!(mir.functions.contains_key(&FunctionId(900)));
    check_calls(&mir, FunctionId(900));
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    assert!(analysis
        .function(runmat_types::ProgramFunctionId(0))
        .is_none());
    assert!(analysis
        .function(runmat_types::ProgramFunctionId(900))
        .is_some());
    let mut futures = 0;
    for fact in analysis
        .program_points
        .iter()
        .flat_map(|point| &point.locals)
    {
        if let Some(runmat_types::ValueFact {
            kind:
                runmat_types::ValueKindFact::Execution(runmat_types::ExecutionFact::Future {
                    output,
                    ..
                }),
            ..
        }) = &fact.fact
        {
            assert_eq!(
                *output,
                runmat_types::ValueSequenceFact::dynamic(),
                "foreign target must not borrow the colliding local function's numeric output"
            );
            futures += 1;
        }
    }
    assert!(futures > 0, "analysis must retain the lazy future fact");
}

fn check_calls(mir: &runmat_mir::MirAssembly, local: FunctionId) {
    let values = mir.bodies.values().flat_map(|body| {
        body.blocks.iter().flat_map(|block| {
            block
                .statements
                .iter()
                .filter_map(|statement| match &statement.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => Some(value),
                    _ => None,
                })
        })
    });
    let mut external = 0;
    let mut internal = 0;
    for value in values {
        match value {
            MirRvalue::ExternalFuture {
                function,
                display_name,
                ..
            } => {
                assert_eq!(*function, FunctionId(0));
                assert_eq!(display_name, "remote");
                external += 1;
            }
            MirRvalue::Call(call) if matches!(call.callee, MirCallee::Static(CallableIdentity::BoundFunction(id)) if id == local) =>
            {
                internal += 1;
            }
            MirRvalue::Future { .. } => {
                panic!("foreign async declaration contaminated local function")
            }
            _ => {}
        }
    }
    assert_eq!((external, internal), (1, 1));
}

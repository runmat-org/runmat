use super::*;

fn lower(source: &str, bindings: &HashMap<String, usize>) -> (HirAssembly, crate::Bytecode) {
    let ast = runmat_parser::parse(source).unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::new(bindings)).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    let bytecode = crate::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap();
    (hir.assembly, bytecode)
}

#[test]
fn assigned_capture_preserves_the_exact_anonymous_forwarding_abi() {
    let source = "eval('shadowed_source = @() 29; shadowed_array = [29, 30];'); pinned = @() shadowed_source(); pinned_complex = @() 1 + shadowed_source(); pinned_array = @() shadowed_array(2);";
    let (original, original_code) = lower(source, &HashMap::new());
    let assigned = HashMap::from([
        ("shadowed_source".to_owned(), 0),
        ("shadowed_array".to_owned(), 1),
    ]);
    let (selected, selected_code) = lower(source, &assigned);
    for function in original
        .functions
        .iter()
        .filter(|function| function.kind == FunctionKind::Anonymous)
    {
        let original_body = original_code
            .function_registry
            .get_current(function.id)
            .unwrap();
        let selected_body = selected_code
            .function_registry
            .get_current(function.id)
            .unwrap();
        let abi = ClosureIssuanceAbi::capture(&original, original_body, function.id).unwrap();
        assert!(abi.canonical_output_forwarding);
        let plan = abi
            .capture_plan(&selected, selected_body, &assigned)
            .expect("assigned workspace capture retains its compiler-owned output forwarding pair");
        assert_eq!(plan.len(), 1);
        assert!(
            matches!(&plan[0], SelectedClosureCapture::AssignedWorkspace { name, slot } if assigned.get(name) == Some(slot))
        );
        // Matching presence/count alone is not sufficient: both the original
        // and the selected destination must remain owned by this exact ABI.
        let mut changed = selected.clone();
        changed
            .functions
            .iter_mut()
            .find(|candidate| candidate.id == function.id)
            .unwrap()
            .abi
            .implicit_nargout = None;
        assert!(abi
            .capture_plan(&changed, selected_body, &assigned)
            .is_none());
        let mut changed = selected.clone();
        let changed_function = changed
            .functions
            .iter_mut()
            .find(|candidate| candidate.id == function.id)
            .unwrap();
        let count_binding = changed_function.abi.implicit_nargout.unwrap();
        let runmat_hir::HirStmtKind::SequenceAssign { target, .. } =
            &mut changed_function.body.statements[0].kind
        else {
            panic!("canonical forwarding body")
        };
        let runmat_hir::HirSequenceTarget::CellContents { base, .. } = target else {
            panic!("canonical destination")
        };
        base.kind = runmat_hir::HirExprKind::Binding(count_binding);
        assert!(abi
            .capture_plan(&changed, selected_body, &assigned)
            .is_none());
        assert!(abi
            .capture_plan(&selected, selected_body, &HashMap::new())
            .is_none());
    }
}

use super::*;
use runmat_execution::value::{CallableValue, GraphBindingKind, GraphNode, GraphProgramUnitId};
use runmat_execution::{Digest, ExecutableSourceRecordV2, FunctionBindingOrigin};
use runmat_types::{BindingId, FunctionSourceRole, ProgramFunctionId, ProgramSourceId};
use runmat_value::{BoundFunctionBinding, CellArray, Closure};
use std::rc::Rc;

fn origin(function: u32) -> FunctionBindingOrigin {
    FunctionBindingOrigin {
        program: Digest::sha256(b"graph-codec-test"),
        function: ProgramFunctionId(function),
    }
}

fn owner(function: u32) -> GraphBoundOwner {
    GraphBoundOwner {
        unit: GraphProgramUnitId(0),
        origin: origin(function),
    }
}

fn source_map() -> ExecutableSourceMapV2 {
    ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
        source_id: ProgramSourceId(0),
        owner_identity: "graph-codec-test".into(),
        relative_path: "nested.m".into(),
        content_revision: Digest::sha256(b"function nested"),
        display_name: "nested.m".into(),
        text: "function nested".into(),
    }])
    .unwrap()
}

fn bindings() -> HashMap<GraphBoundOwner, BoundFunctionBinding> {
    HashMap::from([
        (owner(2), BoundFunctionBinding::portable(2, origin(2))),
        (owner(3), BoundFunctionBinding::portable(3, origin(3))),
    ])
}

fn layouts() -> HashMap<GraphBoundOwner, GraphLexicalLayout> {
    HashMap::from([
        (
            owner(1),
            GraphLexicalLayout {
                activation_parent: None,
                bindings: vec![(BindingId(7), GraphBindingKind::Owned)],
            },
        ),
        (
            owner(2),
            GraphLexicalLayout {
                activation_parent: Some(owner(1)),
                bindings: vec![(BindingId(7), GraphBindingKind::Inherited)],
            },
        ),
        (
            owner(3),
            GraphLexicalLayout {
                activation_parent: None,
                bindings: Vec::new(),
            },
        ),
    ])
}

fn proofs() -> HashMap<GraphBoundOwner, GraphCallableProof> {
    HashMap::from([
        (
            owner(2),
            GraphCallableProof {
                qualified_name: "nested".into(),
                source_role: FunctionSourceRole::Nested,
                issued_source_file: None,
                source_file: None,
                captures: GraphCaptureProof::NestedShared {
                    vm_copied: 1,
                    zero_copy: GraphNestedZeroCopy::SharedActivation,
                },
            },
        ),
        (
            owner(3),
            GraphCallableProof {
                qualified_name: "anonymous".into(),
                source_role: FunctionSourceRole::Anonymous,
                issued_source_file: None,
                source_file: None,
                captures: GraphCaptureProof::Exact(1),
            },
        ),
    ])
}

mod cycle;
mod depth;
mod exception;
mod captures;
mod forgery;
mod invocation;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn selected_local_source_path_becomes_portable_and_relocalizes_after_admission() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let mut admitted_layouts = layouts();
        admitted_layouts.insert(
            owner(2),
            GraphLexicalLayout {
                activation_parent: None,
                bindings: Vec::new(),
            },
        );
        let mut admitted_proofs = proofs();
        let proof = admitted_proofs
            .get_mut(&owner(2))
            .expect("selected nested proof");
        proof.issued_source_file = Some("/selected/outer.m".into());
        proof.source_file = Some("outer.m".into());
        proof.captures = GraphCaptureProof::NestedShared {
            vm_copied: 0,
            zero_copy: GraphNestedZeroCopy::RequiresCopies,
        };
        let input = Value::Closure(Closure::new(
            "nested".into(),
            Some(BoundFunctionBinding::portable(2, origin(2))),
            FunctionSourceRole::Nested,
            Some("/selected/outer.m".into()),
            Vec::new(),
            None,
        ));
        let Value::Closure(independent) = input.clone() else {
            unreachable!()
        };
        let independent = Value::Closure(Closure::from_data(independent.into_data()));
        assert_eq!(input, independent, "semantic equality is unchanged");
        assert_ne!(
            input.callable_instance_key(),
            independent.callable_instance_key()
        );
        let graph = encode_payload_graph_in_context(
            &[input.clone(), input.clone(), independent],
            &runtime,
            &source_map(),
            &admitted_layouts,
            &admitted_proofs,
            ValueLimits::default(),
        )
        .expect("local callable is validated before portable projection");
        assert_eq!(
            graph.roots[0], graph.roots[1],
            "one instance has one graph node"
        );
        assert_ne!(
            graph.roots[0], graph.roots[2],
            "independent equal values do not alias"
        );
        let GraphNode::Callable(wire) = &graph.nodes[graph.roots[0].0 as usize] else {
            panic!("callable root")
        };
        assert_eq!(wire.source_file.as_deref(), Some("outer.m"));
        assert_eq!(
            wire.callable_digest,
            CallableValue::identity_digest(
                wire.owner.portable_owner(),
                &wire.qualified_name,
                wire.form,
                wire.source_role,
                Some("outer.m"),
            )
        );
        let rebuilt = decode_payload_graph_in_context(
            &graph,
            &runtime,
            &source_map(),
            &admitted_layouts,
            &admitted_proofs,
            &bindings(),
            ValueLimits::default(),
        )
        .expect("admitted portable callable reissues its selected local path");
        let [Value::Closure(first), Value::Closure(second), Value::Closure(third)] =
            rebuilt.values()
        else {
            panic!("rebuilt closure")
        };
        assert_eq!(first.source_file.as_deref(), Some("/selected/outer.m"));
        assert_eq!(first.instance_id(), second.instance_id());
        assert_ne!(first.instance_id(), third.instance_id());
        assert_ne!(
            rebuilt.values()[0].callable_instance_key(),
            input.callable_instance_key(),
            "reconstructed graph issues a fresh local callable instance"
        );

        let Value::Closure(forged) = input else {
            unreachable!()
        };
        let mut forged = forged.into_data();
        forged.source_file = Some("/other/outer.m".into());
        assert!(encode_payload_graph_in_context(
            &[Value::Closure(Closure::from_data(forged))],
            &runtime,
            &source_map(),
            &admitted_layouts,
            &admitted_proofs,
            ValueLimits::default(),
        )
        .is_err());
    });
}

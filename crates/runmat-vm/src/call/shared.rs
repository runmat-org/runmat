use runmat_runtime::call::arguments::{ArgumentSpec, MaterializedArgument};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub async fn build_expanded_args_from_specs(
    stack: &mut Vec<Value>,
    specs: &[ArgumentSpec],
) -> Result<Vec<Value>, RuntimeError> {
    let mut arguments = Vec::with_capacity(specs.len());
    for spec in specs.iter().rev() {
        if spec.is_expand {
            let mut indices = Vec::with_capacity(spec.num_indices);
            for _ in 0..spec.num_indices {
                indices.push(stack.pop().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
                })?);
            }
            indices.reverse();
            let base = stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
            })?;
            arguments.push(MaterializedArgument::Expansion {
                base,
                indices,
                expand_all: spec.expand_all,
            });
        } else {
            arguments.push(MaterializedArgument::Single(stack.pop().ok_or_else(
                || crate::interpreter::errors::mex("StackUnderflow", "stack underflow"),
            )?));
        }
    }
    arguments.reverse();
    runmat_runtime::call::arguments::expand_arguments(arguments).await
}

#[cfg(test)]
mod tests {
    use super::build_expanded_args_from_specs;
    use futures::executor::block_on;
    use runmat_hir::{CallableFallbackPolicy, CallableIdentity, FunctionId};
    use runmat_hir::{QualifiedName, SymbolName};
    use runmat_runtime::call::arguments::ArgumentSpec;
    use runmat_runtime::call::identity::{
        external_qualified_display_name, external_qualified_identity,
    };
    use runmat_runtime::indexing::EndExpr;
    use runmat_runtime::object::dispatch::{
        class_defines_member_subsasgn, class_defines_member_subsref,
    };
    use runmat_runtime::object::indexing::{
        build_object_paren_expr_selector_values, build_object_paren_selector_values,
        ObjectIndexDescriptor, ObjectIndexOp, ObjectIndexSelector, ObjectParenExprSelectorSpec,
        OBJECT_END_RANGE_TAG, OBJECT_PROTOCOL_KIND_BRACE, OBJECT_PROTOCOL_KIND_MEMBER,
        OBJECT_PROTOCOL_SUBSASGN, OBJECT_PROTOCOL_SUBSREF, OBJECT_SELECTOR_COLON,
        OBJECT_SELECTOR_END,
    };
    use runmat_types::MemberAccess;
    use runmat_value::{IntValue, Value};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn object_selector_preserves_exact_integer_scalar_class() {
        let selector = Value::Int(IntValue::U64(u64::MAX));
        let values = build_object_paren_selector_values(1, 0, 0, std::slice::from_ref(&selector))
            .expect("object selector");
        assert_eq!(values, vec![selector]);
    }

    macro_rules! build_object_paren_expr_selector_values_from_parts {
        (
            $dims:expr,
            $colon_mask:expr,
            $end_mask:expr,
            $range_dims:expr,
            $range_params:expr,
            $range_start_exprs:expr,
            $range_step_exprs:expr,
            $range_end_exprs:expr,
            $end_numeric_exprs:expr,
            $numeric:expr
            $(,)?
        ) => {
            build_object_paren_expr_selector_values(ObjectParenExprSelectorSpec {
                dims: $dims,
                colon_mask: $colon_mask,
                end_mask: $end_mask,
                range_dims: $range_dims,
                range_params: $range_params,
                range_start_exprs: $range_start_exprs,
                range_step_exprs: $range_step_exprs,
                range_end_exprs: $range_end_exprs,
                end_numeric_exprs: $end_numeric_exprs,
                numeric: $numeric,
            })
        };
    }

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    #[test]
    fn object_index_descriptor_serializes_protocol_args_once() {
        let descriptor = ObjectIndexDescriptor::subsref_brace(
            Value::Num(1.0),
            ObjectIndexSelector::IndexValues {
                values: vec![Value::Num(2.0)],
            },
        );

        let (base, method, args) = descriptor
            .into_method_invocation()
            .expect("descriptor args");
        assert_eq!(base, Value::Num(1.0));
        assert_eq!(method, OBJECT_PROTOCOL_SUBSREF.to_string());
        assert_eq!(
            args[0],
            Value::String(OBJECT_PROTOCOL_KIND_BRACE.to_string())
        );
        match &args[1] {
            Value::Cell(cell) => assert_eq!(cell.data[0].clone(), Value::Num(2.0)),
            other => panic!("expected selector cell, got {other:?}"),
        }
    }

    #[test]
    fn object_member_descriptor_carries_rhs() {
        let descriptor = ObjectIndexDescriptor::member(
            Value::Num(1.0),
            ObjectIndexOp::Subsasgn,
            "field".to_string(),
            Some(Value::Num(9.0)),
        );

        let (base, method, args) = descriptor
            .into_method_invocation()
            .expect("descriptor args");
        assert_eq!(base, Value::Num(1.0));
        assert_eq!(method, OBJECT_PROTOCOL_SUBSASGN.to_string());
        assert_eq!(
            args[0],
            Value::String(OBJECT_PROTOCOL_KIND_MEMBER.to_string())
        );
        assert_eq!(args[1], Value::String("field".to_string()));
        assert_eq!(args[2], Value::Num(9.0));
    }

    #[test]
    fn external_qualified_identity_preserves_malformed_base_segment() {
        let identity = external_qualified_identity("pkg..Point", "origin");
        let CallableIdentity::ExternalName(QualifiedName(segments)) = identity else {
            panic!("expected external qualified identity");
        };
        assert_eq!(
            segments,
            vec![
                SymbolName("pkg..Point".to_string()),
                SymbolName("origin".to_string())
            ]
        );
    }

    #[test]
    fn external_qualified_identity_splits_well_formed_base_segments() {
        let identity = external_qualified_identity("pkg.Point", "origin");
        let CallableIdentity::ExternalName(QualifiedName(segments)) = identity else {
            panic!("expected external qualified identity");
        };
        assert_eq!(
            segments,
            vec![
                SymbolName("pkg".to_string()),
                SymbolName("Point".to_string()),
                SymbolName("origin".to_string())
            ]
        );
    }

    #[test]
    fn external_qualified_display_name_preserves_malformed_base_shape() {
        assert_eq!(
            external_qualified_display_name("pkg..Point", "origin"),
            "pkg..Point.origin"
        );
    }

    #[test]
    fn external_qualified_display_name_renders_well_formed_qualified_name() {
        assert_eq!(
            external_qualified_display_name("pkg.Point", "origin"),
            "pkg.Point.origin"
        );
    }

    #[test]
    fn class_defines_member_subsref_includes_inherited_method_metadata() {
        let parent_name = unique_class_name("vm_subsref_parent");
        let child_name = unique_class_name("vm_subsref_child");
        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            OBJECT_PROTOCOL_SUBSREF.to_string(),
            runmat_runtime::class_registry::RuntimeMethod {
                name: OBJECT_PROTOCOL_SUBSREF.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "subsref_impl".to_string(),
                implicit_class_argument: None,
            },
        );
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: parent_name.clone(),
                parent: None,
                properties: HashMap::new(),
                methods: parent_methods,
            },
        );
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: child_name.clone(),
                parent: Some(parent_name),
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
        );

        let child = runmat_runtime::class_registry::RuntimeClass {
            name: child_name,
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        assert!(class_defines_member_subsref(&child));
    }

    #[test]
    fn class_defines_member_subsasgn_includes_inherited_method_metadata() {
        let parent_name = unique_class_name("vm_subsasgn_parent");
        let child_name = unique_class_name("vm_subsasgn_child");
        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            OBJECT_PROTOCOL_SUBSASGN.to_string(),
            runmat_runtime::class_registry::RuntimeMethod {
                name: OBJECT_PROTOCOL_SUBSASGN.to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "subsasgn_impl".to_string(),
                implicit_class_argument: None,
            },
        );
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: parent_name.clone(),
                parent: None,
                properties: HashMap::new(),
                methods: parent_methods,
            },
        );
        runmat_runtime::class_registry::register_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: child_name.clone(),
                parent: Some(parent_name),
                properties: HashMap::new(),
                methods: HashMap::new(),
            },
        );

        let child = runmat_runtime::class_registry::RuntimeClass {
            name: child_name,
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        assert!(class_defines_member_subsasgn(&child));
    }

    #[test]
    fn object_paren_selector_values_preserve_colon_end_and_numeric_order() {
        let selectors = build_object_paren_selector_values(3, 0b001, 0b010, &[Value::Num(9.0)])
            .expect("selector values");
        assert_eq!(selectors.len(), 3);
        assert_eq!(
            selectors[0],
            Value::String(OBJECT_SELECTOR_COLON.to_string())
        );
        assert_eq!(selectors[1], Value::String(OBJECT_SELECTOR_END.to_string()));
        assert_eq!(selectors[2], Value::Num(9.0));
    }

    #[test]
    fn object_paren_selector_values_validate_numeric_arity() {
        let missing = build_object_paren_selector_values(2, 0, 0, &[Value::Num(1.0)])
            .expect_err("missing selector should fail");
        assert_eq!(missing.identifier(), Some("RunMat:MissingNumericIndex"));

        let extra =
            build_object_paren_selector_values(2, 0b01, 0, &[Value::Num(2.0), Value::Num(3.0)])
                .expect_err("extra selector should fail");
        assert_eq!(extra.identifier(), Some("RunMat:UnexpectedNumericIndex"));
    }

    #[test]
    fn object_paren_selector_values_accept_string_selector() {
        let selectors =
            build_object_paren_selector_values(1, 0, 0, &[Value::String("key".to_string())])
                .expect("string selector should serialize");
        assert_eq!(selectors, vec![Value::String("key".to_string())]);
    }

    #[test]
    fn object_paren_selector_values_reject_unsupported_selector_type() {
        let err = build_object_paren_selector_values(
            1,
            0,
            0,
            &[Value::Struct(runmat_value::StructValue::new())],
        )
        .expect_err("unsupported selector should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:ObjectSelectorTypeUnsupported")
        );
    }

    #[test]
    fn object_paren_selector_values_reject_out_of_bounds_mask_bits() {
        let err = build_object_paren_selector_values(1, 0b10, 0, &[Value::Num(1.0)])
            .expect_err("out-of-bounds selector mask should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidSelectorMaskPlan"));
    }

    #[test]
    fn object_paren_selector_values_reject_overlapping_colon_end_mask_bits() {
        let err = build_object_paren_selector_values(1, 0b1, 0b1, &[])
            .expect_err("overlapping selector mask bits should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidSelectorMaskPlan"));
    }

    #[test]
    fn object_paren_selector_values_support_dims_beyond_mask_width() {
        let numeric: Vec<Value> = (0..31).map(|v| Value::Num((v + 1) as f64)).collect();
        let selectors = build_object_paren_selector_values(33, 0b1, 0b10, &numeric)
            .expect("selector values for dims beyond mask width");
        assert_eq!(selectors.len(), 33);
        assert_eq!(
            selectors[0],
            Value::String(OBJECT_SELECTOR_COLON.to_string())
        );
        assert_eq!(selectors[1], Value::String(OBJECT_SELECTOR_END.to_string()));
        assert_eq!(selectors[32], Value::Num(31.0));
    }

    #[test]
    fn object_paren_expr_selector_values_encode_end_expression_range_descriptors() {
        let selectors = build_object_paren_expr_selector_values_from_parts!(
            2,
            0,
            0,
            &[0],
            &[(1.0, 2.0)],
            &[Some(EndExpr::Sub(
                Box::new(EndExpr::End),
                Box::new(EndExpr::Const(1.0)),
            ))],
            &[None],
            &[EndExpr::End],
            &[],
            &[Value::Num(4.0)],
        )
        .expect("expr selector values");

        assert_eq!(selectors.len(), 2);
        match &selectors[0] {
            Value::Cell(cell) => {
                assert_eq!(
                    cell.data[2].clone(),
                    Value::String(OBJECT_END_RANGE_TAG.to_string())
                );
                assert_eq!(cell.data[1].clone(), Value::Num(2.0));
                assert_eq!(
                    cell.data[3].clone(),
                    Value::String(OBJECT_SELECTOR_END.to_string())
                );
            }
            other => panic!("expected range descriptor cell, got {other:?}"),
        }
        assert_eq!(selectors[1], Value::Num(4.0));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_end_call_without_callable_name() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            1,
            0,
            0,
            &[0],
            &[(1.0, 1.0)],
            &[None],
            &[None],
            &[EndExpr::ResolvedCall {
                identity: CallableIdentity::BoundFunction(FunctionId(7)),
                fallback_policy: CallableFallbackPolicy::None,
                args: vec![],
            }],
            &[],
            &[],
        )
        .expect_err("missing callable name should fail");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_malformed_external_end_call_name() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            1,
            0,
            0,
            &[0],
            &[(1.0, 1.0)],
            &[None],
            &[None],
            &[EndExpr::ResolvedCall {
                identity: CallableIdentity::ExternalName(QualifiedName(vec![
                    SymbolName("pkg".to_string()),
                    SymbolName("".to_string()),
                    SymbolName("remote".to_string()),
                ])),
                fallback_policy: CallableFallbackPolicy::ExternalBoundary,
                args: vec![],
            }],
            &[],
            &[],
        )
        .expect_err("malformed external callable name should fail");
        assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_invalid_range_plan_metadata() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            2,
            0,
            0,
            &[0],
            &[(1.0, 2.0)],
            &[None],
            &[None],
            &[],
            &[],
            &[Value::Num(4.0)],
        )
        .expect_err("inconsistent range metadata should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_range_dim_conflicting_with_colon_mask() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            2,
            0b01,
            0,
            &[0],
            &[(1.0, 2.0)],
            &[None],
            &[None],
            &[EndExpr::End],
            &[],
            &[Value::Num(4.0)],
        )
        .expect_err("range dimension conflicting with colon mask should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_range_dim_conflicting_with_end_mask() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            2,
            0,
            0b01,
            &[0],
            &[(1.0, 2.0)],
            &[None],
            &[None],
            &[EndExpr::End],
            &[],
            &[Value::Num(4.0)],
        )
        .expect_err("range dimension conflicting with end mask should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_out_of_bounds_mask_bits() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            1,
            0b10,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[Value::Num(1.0)],
        )
        .expect_err("out-of-bounds selector mask should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidSelectorMaskPlan"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_overlapping_colon_end_mask_bits() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            1,
            0b1,
            0b1,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[]
        )
        .expect_err("overlapping selector mask bits should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidSelectorMaskPlan"));
    }

    #[test]
    fn object_paren_expr_selector_values_support_dims_beyond_mask_width() {
        let numeric: Vec<Value> = (0..32).map(|v| Value::Num((v + 1) as f64)).collect();
        let selectors = build_object_paren_expr_selector_values_from_parts!(
            33,
            0,
            0,
            &[32],
            &[(1.0, 1.0)],
            &[None],
            &[None],
            &[EndExpr::End],
            &[],
            &numeric,
        )
        .expect("expr selector values for dims beyond mask width");

        assert_eq!(selectors.len(), 33);
        assert_eq!(selectors[31], Value::Num(32.0));
        match &selectors[32] {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 4);
            }
            other => panic!("expected range descriptor cell, got {other:?}"),
        }
    }

    #[test]
    fn object_paren_expr_selector_values_reject_duplicate_range_dims() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            2,
            0,
            0,
            &[0, 0],
            &[(1.0, 1.0), (2.0, 1.0)],
            &[None, None],
            &[None, None],
            &[EndExpr::End, EndExpr::End],
            &[],
            &[],
        )
        .expect_err("duplicate range dimensions should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_out_of_bounds_range_dim() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            1,
            0,
            0,
            &[1],
            &[(1.0, 1.0)],
            &[None],
            &[None],
            &[EndExpr::End],
            &[],
            &[],
        )
        .expect_err("out-of-bounds range dimension should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorDim"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_unsupported_numeric_selector_type() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            1,
            0,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[Value::Struct(runmat_value::StructValue::new())],
        )
        .expect_err("unsupported object selector type should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:ObjectSelectorTypeUnsupported")
        );
    }

    #[test]
    fn object_paren_expr_selector_values_accept_string_selector_in_mixed_plan() {
        let selectors = build_object_paren_expr_selector_values_from_parts!(
            2,
            0,
            0,
            &[0],
            &[(1.0, 1.0)],
            &[None],
            &[None],
            &[EndExpr::End],
            &[],
            &[Value::String("key".to_string())],
        )
        .expect("mixed string selector should serialize");
        assert_eq!(selectors.len(), 2);
        assert_eq!(selectors[1], Value::String("key".to_string()));
    }

    #[test]
    fn object_paren_expr_selector_values_accept_cell_selector_in_mixed_plan() {
        let key_cell = runmat_value::CellArray::new(vec![Value::String("k".to_string())], 1, 1)
            .expect("key cell");
        let selectors = build_object_paren_expr_selector_values_from_parts!(
            2,
            0,
            0,
            &[0],
            &[(1.0, 1.0)],
            &[None],
            &[None],
            &[EndExpr::End],
            &[],
            &[Value::Cell(key_cell.clone())],
        )
        .expect("mixed cell selector should serialize");
        assert_eq!(selectors.len(), 2);
        assert_eq!(selectors[1], Value::Cell(key_cell));
    }

    #[test]
    fn object_paren_expr_selector_values_encode_numeric_end_expressions() {
        let selectors = build_object_paren_expr_selector_values_from_parts!(
            1,
            0,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[(
                0,
                EndExpr::Div(Box::new(EndExpr::End), Box::new(EndExpr::Const(2.0))),
            )],
            &[Value::Num(0.0)],
        )
        .expect("numeric end expression selector should serialize");
        assert_eq!(selectors.len(), 1);
        match &selectors[0] {
            Value::Cell(cell) => {
                assert_eq!(cell.data[0].clone(), Value::String("/".to_string()));
            }
            other => panic!("expected encoded end expression cell, got {other:?}"),
        }
    }

    #[test]
    fn object_paren_expr_selector_values_reject_duplicate_numeric_end_expr_positions() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            1,
            0,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[
                (
                    0,
                    EndExpr::Div(Box::new(EndExpr::End), Box::new(EndExpr::Const(2.0))),
                ),
                (
                    0,
                    EndExpr::Sub(Box::new(EndExpr::End), Box::new(EndExpr::Const(1.0))),
                ),
            ],
            &[Value::Num(0.0)],
        )
        .expect_err("duplicate numeric end-expression positions should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidEndSelectorPlan"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_out_of_bounds_numeric_end_expr_positions() {
        let err = build_object_paren_expr_selector_values_from_parts!(
            1,
            0,
            0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[(
                1,
                EndExpr::Div(Box::new(EndExpr::End), Box::new(EndExpr::Const(2.0))),
            )],
            &[Value::Num(0.0)],
        )
        .expect_err("out-of-bounds numeric end-expression position should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidEndSelectorPlan"));
    }

    #[test]
    fn build_expanded_args_from_specs_supports_output_list_index_expansion() {
        let mut stack = vec![
            Value::OutputList(vec![Value::Num(9.0), Value::Num(2.0)]),
            Value::Num(1.0),
        ];
        let specs = vec![ArgumentSpec {
            is_expand: true,
            num_indices: 1,
            expand_all: false,
        }];
        let expanded =
            block_on(build_expanded_args_from_specs(&mut stack, &specs)).expect("expanded args");
        assert_eq!(expanded, vec![Value::Num(9.0)]);

        let mut stack = vec![
            Value::OutputList(vec![Value::Num(9.0), Value::Num(2.0)]),
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ];
        let expanded =
            block_on(build_expanded_args_from_specs(&mut stack, &specs)).expect("expanded args");
        assert_eq!(expanded, vec![Value::Num(9.0), Value::Num(2.0)]);
    }
}

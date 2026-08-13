use runmat_runtime::call::arguments::ArgumentSpec;
use runmat_runtime::object::dispatch::call_object_index_descriptor_method_with_outputs;
use runmat_runtime::object::indexing::{ObjectIndexDescriptor, ObjectIndexSelector};
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::Value;
use std::future::Future;

pub fn expand_cell_indices(
    cell: &runmat_value::CellArray,
    indices: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    runmat_runtime::object::cell::expand_cell_indices(cell, indices)
}

pub fn expand_all_cell(cell: &runmat_value::CellArray) -> Result<Vec<Value>, RuntimeError> {
    runmat_runtime::object::cell::expand_all_cell_values(cell)
}

pub(crate) async fn expand_brace_values(
    base: Value,
    raw_indices: &[Value],
    pad_to_outputs: Option<usize>,
) -> Result<Vec<Value>, RuntimeError> {
    async fn expand_object_brace_values(
        base: Value,
        raw_indices: &[Value],
        pad_to_outputs: Option<usize>,
    ) -> Result<Vec<Value>, RuntimeError> {
        let value = call_object_index_descriptor_method_with_outputs(
            ObjectIndexDescriptor::subsref_brace(
                base,
                ObjectIndexSelector::IndexValues {
                    values: raw_indices.to_vec(),
                },
            ),
            pad_to_outputs.unwrap_or(1),
        )
        .await?;
        Ok(match value {
            Value::OutputList(values) => values,
            other => vec![other],
        })
    }

    let mut values = match base {
        Value::Cell(ca) => {
            if raw_indices.is_empty() {
                if let Some(out_count) = pad_to_outputs {
                    runmat_runtime::object::cell::expand_cell_values(&ca, &[], out_count)?
                } else {
                    expand_all_cell(&ca)?
                }
            } else {
                expand_cell_indices(&ca, raw_indices)?
            }
        }
        Value::Object(obj) => {
            expand_object_brace_values(Value::Object(obj), raw_indices, pad_to_outputs).await?
        }
        Value::HandleObject(handle) => {
            expand_object_brace_values(Value::HandleObject(handle), raw_indices, pad_to_outputs)
                .await?
        }
        _ => {
            return Err(crate::interpreter::errors::mex(
                "CellExpansionOnNonCell",
                "Cell expansion on non-cell",
            ))
        }
    };
    if let Some(out_count) = pad_to_outputs {
        if values.len() > out_count {
            values.truncate(out_count);
        } else {
            values.resize(out_count, Value::Num(0.0));
        }
    }
    Ok(values)
}

pub async fn build_expanded_args_from_specs<ExpandObjectAll, ExpandObjectIndices, FutAll, FutIdx>(
    stack: &mut Vec<Value>,
    specs: &[ArgumentSpec],
    invalid_expand_all_msg: &str,
    invalid_expand_msg: &str,
    mut expand_object_all: ExpandObjectAll,
    mut expand_object_indices: ExpandObjectIndices,
) -> Result<Vec<Value>, RuntimeError>
where
    ExpandObjectAll: FnMut(Value) -> FutAll,
    ExpandObjectIndices: FnMut(Value, Vec<Value>) -> FutIdx,
    FutAll: Future<Output = Result<Vec<Value>, RuntimeError>>,
    FutIdx: Future<Output = Result<Vec<Value>, RuntimeError>>,
{
    let mut temp: Vec<Value> = Vec::new();
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

            let expanded = if spec.expand_all {
                match base {
                    Value::OutputList(outputs) => outputs,
                    Value::Cell(ca) => expand_all_cell(&ca)?,
                    other @ Value::Object(_) | other @ Value::HandleObject(_) => {
                        expand_object_all(other).await?
                    }
                    _ => {
                        return Err(crate::interpreter::errors::mex(
                            "InvalidExpandAllTarget",
                            invalid_expand_all_msg,
                        ))
                    }
                }
            } else {
                match (base, indices.len()) {
                    (Value::Cell(ca), 1) | (Value::Cell(ca), 2) => {
                        expand_cell_indices(&ca, &indices)?
                    }
                    (Value::OutputList(outputs), 1) | (Value::OutputList(outputs), 2) => {
                        let cols = outputs.len();
                        let cell =
                            build_cell_array_with_shape(outputs, 1, cols, "output-list expansion")?;
                        expand_cell_indices(&cell, &indices)?
                    }
                    (other @ Value::Object(_), _) | (other @ Value::HandleObject(_), _) => {
                        expand_object_indices(other, indices).await?
                    }
                    _ => {
                        return Err(crate::interpreter::errors::mex(
                            "InvalidExpandTarget",
                            invalid_expand_msg,
                        ))
                    }
                }
            };
            temp.extend(expanded.into_iter().rev());
        } else {
            temp.push(stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
            })?);
        }
    }
    temp.reverse();
    Ok(temp)
}

fn build_cell_array_with_shape(
    values: Vec<Value>,
    rows: usize,
    cols: usize,
    context: &str,
) -> Result<runmat_value::CellArray, RuntimeError> {
    runmat_value::CellArray::new(values, rows, cols).map_err(|e| {
        build_runtime_error(format!("{context}: {e}"))
            .with_identifier("RunMat:ShapeMismatch")
            .build()
    })
}

#[cfg(test)]
mod tests {
    use super::{build_expanded_args_from_specs, ObjectIndexDescriptor, ObjectIndexSelector};
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
        build_object_paren_expr_selector_values, build_object_paren_selector_values, ObjectIndexOp,
        ObjectParenExprSelectorSpec, OBJECT_END_RANGE_TAG, OBJECT_PROTOCOL_KIND_BRACE,
        OBJECT_PROTOCOL_KIND_MEMBER, OBJECT_PROTOCOL_SUBSASGN, OBJECT_PROTOCOL_SUBSREF,
        OBJECT_SELECTOR_COLON, OBJECT_SELECTOR_END,
    };
    use runmat_types::MemberAccess;
    use runmat_value::{HandleRef, IntValue, Value};
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
    fn cell_builder_maps_shape_errors_to_identifier() {
        let err = super::build_cell_array_with_shape(vec![Value::Num(1.0)], 2, 2, "test")
            .expect_err("expected shape mismatch");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
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
    fn build_expanded_args_from_specs_accepts_handle_object_expansion() {
        let target = runmat_gc::gc_allocate(Value::Num(7.0)).expect("handle target");
        let handle = HandleRef {
            class_name: "HandleThing".to_string(),
            target,
            valid: true,
        };
        let mut stack = vec![Value::HandleObject(handle)];
        let specs = vec![ArgumentSpec {
            is_expand: true,
            num_indices: 0,
            expand_all: true,
        }];
        let expanded = block_on(build_expanded_args_from_specs(
            &mut stack,
            &specs,
            "expand-all failed",
            "expand-indices failed",
            |base| async move {
                match base {
                    Value::HandleObject(_) => Ok(vec![Value::Num(42.0)]),
                    other => panic!("expected handle object expansion path, got {other:?}"),
                }
            },
            |_base, _indices| async move { Ok(vec![]) },
        ))
        .expect("expanded args");
        assert_eq!(expanded, vec![Value::Num(42.0)]);
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
        let expanded = block_on(build_expanded_args_from_specs(
            &mut stack,
            &specs,
            "expand-all failed",
            "expand-indices failed",
            |_base| async move { panic!("unexpected object expand-all path") },
            |_base, _indices| async move { panic!("unexpected object expand-indices path") },
        ))
        .expect("expanded args");
        assert_eq!(expanded, vec![Value::Num(9.0)]);

        let mut stack = vec![
            Value::OutputList(vec![Value::Num(9.0), Value::Num(2.0)]),
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        ];
        let expanded = block_on(build_expanded_args_from_specs(
            &mut stack,
            &specs,
            "expand-all failed",
            "expand-indices failed",
            |_base| async move { panic!("unexpected object expand-all path") },
            |_base, _indices| async move { panic!("unexpected object expand-indices path") },
        ))
        .expect("expanded args");
        assert_eq!(expanded, vec![Value::Num(9.0), Value::Num(2.0)]);
    }
}

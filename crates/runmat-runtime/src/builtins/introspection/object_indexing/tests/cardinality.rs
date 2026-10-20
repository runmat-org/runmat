use super::*;
#[test]
fn default_cell_brace_indexing_counts_selected_outputs() {
    let target = cell(vec![
        Value::String("one".to_string().into()),
        Value::Num(2.0),
        Value::String("three".to_string().into()),
    ]);
    let indices = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("indices");
    let descriptor = substruct(
        crate::OBJECT_INDEX_BRACE,
        cell(vec![Value::Tensor(indices)]),
    );
    assert_eq!(
        call(
            target,
            descriptor,
            Value::String("Statement".to_string().into())
        ),
        Value::Num(2.0)
    );
}

#[test]
fn default_cell_brace_colon_counts_target_elements_without_gather() {
    let target = cell(vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)]);
    let descriptor = substruct(
        crate::OBJECT_INDEX_BRACE,
        cell(vec![Value::String(":".to_string().into())]),
    );
    assert_eq!(
        call(
            target,
            descriptor,
            Value::String(
                "matlab.indexing.IndexingContext.Expression"
                    .to_string()
                    .into()
            ),
        ),
        Value::Num(3.0)
    );
}

#[test]
fn default_cell_brace_multidimensional_colons_use_shape_extents() {
    let target = Value::Cell(
        CellArray::new_with_shape(
            vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
                Value::Num(5.0),
                Value::Num(6.0),
            ],
            vec![2, 3],
        )
        .expect("cell"),
    );
    let descriptor = substruct(
        crate::OBJECT_INDEX_BRACE,
        cell(vec![
            Value::String(":".to_string().into()),
            Value::String(":".to_string().into()),
        ]),
    );
    assert_eq!(
        call(
            target,
            descriptor,
            Value::String("Statement".to_string().into())
        ),
        Value::Num(6.0)
    );
}

#[test]
fn default_cell_brace_empty_logical_selection_can_return_zero() {
    let target = cell(vec![Value::Num(1.0), Value::Num(2.0)]);
    let descriptor = substruct(crate::OBJECT_INDEX_BRACE, cell(vec![Value::Bool(false)]));
    assert_eq!(
        call(
            target,
            descriptor,
            Value::String("Statement".to_string().into())
        ),
        Value::Num(0.0)
    );
}

#[test]
fn default_gpu_brace_colon_counts_shape_without_gather() {
    let target = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 3],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("valid unregistered resident fixture")
            .handle,
    );
    let descriptor = substruct(
        crate::OBJECT_INDEX_BRACE,
        cell(vec![
            Value::String(":".to_string().into()),
            Value::String(":".to_string().into()),
        ]),
    );
    assert_eq!(
        call(
            target,
            descriptor,
            Value::String("Statement".to_string().into())
        ),
        Value::Num(6.0)
    );
}

#[test]
fn enum_context_object_is_accepted() {
    ensure_indexing_context_classes_registered();
    let target = cell(vec![Value::Num(1.0)]);
    let descriptor = substruct(crate::OBJECT_INDEX_PAREN, cell(vec![Value::Num(1.0)]));
    let mut context = runmat_value::UnboundedObjectBuilder::new(INDEXING_CONTEXT_CLASS);
    context.insert(
        ENUM_MEMBER_PROPERTY.to_string(),
        Value::String("Assignment".to_string().into()),
    );
    assert_eq!(
        call(target, descriptor, Value::Object(context.finish())),
        Value::Num(1.0)
    );
}

#[test]
fn object_overload_dispatches_num_arguments_from_subscript_method() {
    block_on(crate::register_test_classes_builtin()).expect("test classes");
    let mut object = runmat_value::UnboundedObjectBuilder::new("OverIdx");
    object.insert("nargs".to_string(), Value::Num(5.0));
    let descriptor = substruct(
        crate::OBJECT_INDEX_MEMBER,
        Value::String("field".to_string().into()),
    );
    assert_eq!(
        call(
            Value::Object(object.finish()),
            descriptor,
            Value::String("Expression".to_string().into()),
        ),
        Value::Num(5.0)
    );
}

#[test]
fn object_overload_uses_inherited_remapped_method_metadata() {
    block_on(crate::register_test_classes_builtin()).expect("test classes");
    let base = "NumArgsFromSubscriptBase";
    let child = "NumArgsFromSubscriptChild";
    let mut methods = HashMap::new();
    methods.insert(
        NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD.into(),
        crate::class_registry::RuntimeMethod {
            name: NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD.into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: format!("OverIdx.{NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD}"),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: base.into(),
        parent: None,
        properties: HashMap::new(),
        methods,
    });
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: child.into(),
        parent: Some(base.into()),
        properties: HashMap::new(),
        methods: HashMap::new(),
    });

    let mut object = runmat_value::UnboundedObjectBuilder::new(child);
    object.insert("nargs".to_string(), Value::Num(7.0));
    let descriptor = substruct(
        crate::OBJECT_INDEX_MEMBER,
        Value::String("field".to_string().into()),
    );
    assert_eq!(
        call(
            Value::Object(object.finish()),
            descriptor,
            Value::String("Expression".to_string().into()),
        ),
        Value::Num(7.0)
    );
}

//! Actual GC roots/generations, never a global expression or variable-name map.
use super::*;
use runmat_gc::GcHandle;

pub(super) struct OwnedNode {
    object: ObjectInstance,
    _root: runmat_gc::ExplicitRoot,
}
impl OwnedNode {
    pub(super) fn read(handle: GcHandle, policy: Policy) -> BuiltinResult<Self> {
        let root = runmat_gc::gc_root(handle)
            .map_err(|source| policy.gc("failed to retain expression node", source))?;
        let object = runmat_gc::gc_with_value(&handle, |value| match value {
            Value::Object(object)
                if object.class_name().is(standard::OPTIMIZATION_EXPRESSION)
                    && object.selected_class().is_none()
                    && matches!(object.optimization_model_payload(), Some(Payload::Node(_))) =>
            {
                Some(object.clone())
            }
            _ => None,
        })
        .map_err(|source| policy.gc("failed to read expression node", source))?
        .ok_or_else(|| {
            policy.invalid("expression edge does not identify an immutable native node")
        })?;
        Ok(Self {
            object,
            _root: root,
        })
    }
    pub(super) fn dependency_at(&self, index: usize) -> Option<GcHandle> {
        self.object
            .optimization_model_payload()
            .and_then(|payload| payload.handle_at(index))
    }
    pub(super) fn node(&self) -> &Node {
        let Some(Payload::Node(node)) = self.object.optimization_model_payload() else {
            unreachable!("validated native node")
        };
        node
    }
}
pub(super) fn reference(value: &Value, policy: Policy) -> BuiltinResult<GcHandle> {
    let Value::Object(object) = value else {
        return Err(policy.invalid("expected an optimization expression"));
    };
    if !is_expression(object) {
        return Err(policy.invalid("expected an original native expression"));
    }
    let Some(Payload::Expression(value)) = object.optimization_model_payload() else {
        unreachable!()
    };
    let node = OwnedNode::read(value.node, policy)?;
    if node.node().shape != value.shape {
        return Err(policy.invalid("expression shape projection differs from its original node"));
    }
    Ok(value.node)
}
pub(super) async fn publish(
    node: Node,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    registration::ensure(
        standard::OPTIMIZATION_EXPRESSION,
        policy,
        cooperation.runtime(),
    )?;
    let shape = node.shape.clone();
    let source = Value::Object(
        ObjectInstance::new(standard::OPTIMIZATION_EXPRESSION.owned())
            .with_optimization_model_payload(Payload::Node(node)),
    );
    // Root the graph before even metadata validation can yield: the caller
    // transfers this node, not a separately retained copy of its child roots.
    let inputs = policy.roots(std::iter::once(&source), cooperation.runtime())?;
    let Value::Object(object) = &source else {
        unreachable!()
    };
    let Some(Payload::Node(node)) = object.optimization_model_payload() else {
        unreachable!()
    };
    if node.shape.len() < 2 || node.index_names.len() != node.shape.len() {
        return Err(policy.invalid("expression shape and axis metadata ranks differ"));
    }
    let mut count = 1usize;
    for (names, extent) in node.index_names.iter().zip(node.shape.iter()) {
        count = count
            .checked_mul(*extent)
            .ok_or_else(|| policy.invalid("expression shape overflows usize"))?;
        if names.as_ref().is_some_and(|names| names.len() != *extent) {
            return Err(policy.invalid("expression index names differ from their axis extent"));
        }
        cooperation.completed(1).await?;
    }
    cooperation.completed(0).await?;
    let root = runmat_gc::gc_allocate_rooted(source)
        .map_err(|source| policy.gc("expression node allocation failed", source))?;
    let public = ObjectInstance::new(standard::OPTIMIZATION_EXPRESSION.owned())
        .with_optimization_model_payload(Payload::Expression(ExpressionRef {
            node: root.handle(),
            shape,
        }));
    let output = RootedValue::in_optional_context(Value::Object(public), cooperation.runtime())?;
    drop(inputs);
    Ok(output)
}
pub(super) async fn as_expression(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if let Value::Object(object) = value {
        if is_expression(object) {
            let _checked = OwnedNode::read(reference(value, policy)?, policy)?;
            return RootedValue::in_optional_context(value.clone(), cooperation.runtime());
        }
        if super::super::optimvar::is_reference(object) {
            let view = super::super::optimvar::View::new(object, cooperation).await?;
            let shape = view.selection().shape().clone();
            let index_names = super::properties::variable_names(&view, policy, cooperation).await?;
            return publish(
                Node {
                    shape,
                    index_names,
                    kind: Kind::Variable(value.clone()),
                },
                policy,
                cooperation,
            )
            .await;
        }
        return Err(policy.invalid("an unrelated object cannot become an optimization expression"));
    }
    let normalized = super::construct::normalize(value, policy, cooperation).await?;
    super::construct::finite_constant(normalized.value(), policy, cooperation).await?;
    let shape = super::construct::numeric_shape(normalized.value(), policy, cooperation).await?;
    let index_names = super::construct::unnamed(shape.len(), policy)?;
    publish(
        Node {
            shape,
            index_names,
            kind: Kind::Constant(normalized.value().clone()),
        },
        policy,
        cooperation,
    )
    .await
}

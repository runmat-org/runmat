use crate::builtins::introspection::object_indexing::{
    brace_assignment_cardinality, member_assignment_cardinality,
};
use crate::execution::{RootedValue, RootedValueLease};
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_gc::{Trace, Tracer};
use runmat_value::Value;

#[derive(Debug, Clone)]
pub enum SequenceEndpointSpec {
    Member(runmat_types::MemberName),
    CellContents(Vec<ObjectIndexComponent>),
}

#[derive(Debug, Clone)]
pub struct PreparedSequenceEndpoint {
    spec: SequenceEndpointSpec,
    cardinality: usize,
    _retained_roots: RootedValueLease,
    _source_roots: RootedValueLease,
}

impl PreparedSequenceEndpoint {
    pub fn visit_retained_values(&self, visitor: &mut impl FnMut(&Value)) {
        self.spec.visit_retained_values(visitor);
    }
    pub async fn prepare(base: &Value, spec: SequenceEndpointSpec) -> Result<Self, RuntimeError> {
        let _inputs =
            RootedValueLease::for_traced(&spec).map_err(|error| error.into_runtime_error())?;
        let _base = RootedValueLease::for_values(std::iter::once(base))
            .map_err(|error| error.into_runtime_error())?;
        let cardinality = match &spec {
            SequenceEndpointSpec::Member(member) => {
                member_assignment_cardinality(base, member.0.clone()).await?
            }
            SequenceEndpointSpec::CellContents(indices) => {
                brace_assignment_cardinality(
                    base,
                    indices
                        .iter()
                        .map(ObjectIndexComponent::protocol_value)
                        .collect(),
                )
                .await?
            }
        };
        let retained_roots =
            RootedValueLease::for_traced(&spec).map_err(|error| error.into_runtime_error())?;
        Ok(Self {
            spec,
            cardinality,
            _retained_roots: retained_roots,
            _source_roots: _base,
        })
    }

    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    pub async fn assign(
        self,
        base: Value,
        values: Vec<Value>,
        caller_function_name: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        self.assign_with_context(None, base, values, caller_function_name)
            .await
    }

    pub async fn assign_in_context(
        self,
        context: &crate::context::RuntimeContext,
        base: Value,
        values: Vec<Value>,
        caller_function_name: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        self.assign_with_context(Some(context), base, values, caller_function_name)
            .await
    }

    async fn assign_with_context(
        self,
        context: Option<&crate::context::RuntimeContext>,
        base: Value,
        values: Vec<Value>,
        caller_function_name: Option<&str>,
    ) -> Result<RootedValue, RuntimeError> {
        let Self {
            spec,
            cardinality,
            _retained_roots,
            _source_roots,
        } = self;
        let _inputs = RootedValueLease::for_values(std::iter::once(&base).chain(values.iter()))
            .map_err(|error| error.into_runtime_error())?;
        if values.len() != cardinality {
            return Err(semantic_error(
                "RunMat:CommaSeparatedListAssignmentArity",
                format!(
                    "sequence assignment requires {} values, but received {}",
                    cardinality,
                    values.len()
                ),
            ));
        }
        match spec {
            SequenceEndpointSpec::Member(field) => match context {
                Some(context) => {
                    crate::object::resolve::store_member_sequence_traced_in_context(
                        context,
                        base,
                        field.0,
                        values,
                        caller_function_name,
                    )
                    .await
                }
                None => {
                    crate::object::resolve::store_member_sequence_traced(
                        base,
                        field.0,
                        values,
                        caller_function_name,
                    )
                    .await
                }
            },
            SequenceEndpointSpec::CellContents(indices) => {
                assign_cell_contents_with_context(context, base, indices, values).await
            }
        }
    }
}

impl SequenceEndpointSpec {
    pub(crate) fn visit_retained_values(&self, visitor: &mut impl FnMut(&Value)) {
        match self {
            Self::Member(_) => {}
            Self::CellContents(components) => {
                for component in components {
                    if let ObjectIndexComponent::Value(value) = component {
                        visitor(value);
                    }
                }
            }
        }
    }
}

impl Trace for SequenceEndpointSpec {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.visit_retained_values(&mut |value| value.trace(tracer));
    }
}

pub(crate) async fn assign_cell_contents_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    indices: Vec<ObjectIndexComponent>,
    values: Vec<Value>,
) -> Result<RootedValue, RuntimeError> {
    let _inputs = RootedValueLease::for_values(std::iter::once(&base).chain(values.iter()))
        .map_err(|error| error.into_runtime_error())?;
    match base {
        Value::Cell(cell) => {
            let positions = if matches!(indices.as_slice(), [ObjectIndexComponent::Colon]) {
                (1..=cell.data.len()).collect::<Vec<_>>()
            } else {
                crate::object::cell::resolve_cell_assignment_positions(
                    &cell,
                    &indices
                        .iter()
                        .map(ObjectIndexComponent::protocol_value)
                        .collect::<Vec<_>>(),
                )?
            };
            crate::object::cell::assign_cell_value_multi(
                cell,
                &positions,
                &values,
                runmat_gc::gc_record_write,
            )
            .and_then(|value| RootedValue::new(value).map_err(|error| error.into_runtime_error()))
        }
        base @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
            let resolution = crate::object::dispatch::resolve_object_index_protocol(
                &base,
                crate::object::indexing::ObjectIndexOp::Subsasgn,
                None,
            )?;
            let Some(method) = resolution else {
                return Err(semantic_error(
                    "MissingObjectProtocol",
                    "brace assignment object does not define subsasgn",
                ));
            };
            let path = ObjectSubscriptPath::single(ObjectSubscript::braces(
                ObjectIndexSelector::IndexValues {
                    components: indices,
                },
            ));
            match context {
                Some(context) => {
                    crate::object::protocol::invoke_resolved_object_assignment_in_context(
                        context, &method, base, path, values,
                    )
                    .await
                }
                None => {
                    crate::object::protocol::invoke_resolved_object_assignment(
                        &method, base, path, values,
                    )
                    .await
                }
            }
        }
        _ => Err(semantic_error(
            "CellAssignmentOnNonCell",
            "brace sequence assignment requires a cell array or indexing object",
        )),
    }
}

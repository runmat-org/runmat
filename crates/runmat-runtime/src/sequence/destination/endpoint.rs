use crate::builtins::introspection::object_indexing::{
    brace_assignment_cardinality, member_assignment_cardinality,
};
use crate::object::indexing::{
    ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
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
}

impl PreparedSequenceEndpoint {
    pub async fn prepare(base: &Value, spec: SequenceEndpointSpec) -> Result<Self, RuntimeError> {
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
        Ok(Self { spec, cardinality })
    }

    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    pub async fn assign(
        self,
        base: Value,
        values: Vec<Value>,
        caller_function_name: Option<&str>,
    ) -> Result<Value, RuntimeError> {
        if values.len() != self.cardinality {
            return Err(semantic_error(
                "RunMat:CommaSeparatedListAssignmentArity",
                format!(
                    "sequence assignment requires {} values, but received {}",
                    self.cardinality,
                    values.len()
                ),
            ));
        }
        match self.spec {
            SequenceEndpointSpec::Member(field) => {
                crate::object::resolve::store_member_sequence_traced(
                    base,
                    field.0,
                    values,
                    caller_function_name,
                )
                .await
            }
            SequenceEndpointSpec::CellContents(indices) => {
                assign_cell_contents(base, indices, values).await
            }
        }
    }
}

pub(crate) async fn assign_cell_contents(
    base: Value,
    indices: Vec<ObjectIndexComponent>,
    values: Vec<Value>,
) -> Result<Value, RuntimeError> {
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
        }
        base @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
            let resolution = crate::object::dispatch::resolve_object_index_protocol(
                &base,
                crate::object::indexing::ObjectIndexOp::Subsasgn,
                None,
            )?;
            let crate::object::protocol::ProtocolResolution::Method(method) = resolution else {
                return Err(semantic_error(
                    "MissingObjectProtocol",
                    "brace assignment object does not define subsasgn",
                ));
            };
            crate::object::protocol::invoke_resolved_object_assignment(
                &method,
                base,
                ObjectSubscriptPath::single(ObjectSubscript::braces(
                    ObjectIndexSelector::IndexValues {
                        components: indices,
                    },
                )),
                values,
            )
            .await
        }
        _ => Err(semantic_error(
            "CellAssignmentOnNonCell",
            "brace sequence assignment requires a cell array or indexing object",
        )),
    }
}

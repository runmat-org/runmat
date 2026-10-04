//! Complete-call selection for a sealed Dataset-family receiver.

use crate::context::host_effect::{
    select_host_effect, DatasetHandleInput, DatasetHandleKey, DatasetHandleOperation,
    DatasetHandleRole, HostEffectOutput, HostEffectReceipt, HostEffectRequest, HostEffectWorkload,
};
use crate::context::RuntimeContext;
use crate::data::authority::{AuthorizedHandle, HandleRole};
use crate::{build_runtime_error, BuiltinResult};
use runmat_execution::Digest;
use runmat_value::{NumericScalar, Value};
use sha2::{Digest as _, Sha256};
use std::future::Future;

pub(super) fn select(
    runtime: Option<&RuntimeContext>,
    owner: &AuthorizedHandle,
    operation: DatasetHandleOperation,
    arguments: &[&Value],
) -> BuiltinResult<Option<HostEffectReceipt>> {
    let facts = arguments
        .iter()
        .map(|&value| input(value))
        .collect::<Vec<_>>();
    select_facts(runtime, owner, operation, &facts)
}

fn select_facts(
    runtime: Option<&RuntimeContext>,
    owner: &AuthorizedHandle,
    operation: DatasetHandleOperation,
    facts: &[DatasetHandleInput],
) -> BuiltinResult<Option<HostEffectReceipt>> {
    let role = match &owner.role {
        HandleRole::Dataset => DatasetHandleRole::Dataset,
        HandleRole::Array(name) => DatasetHandleRole::Array(name),
        HandleRole::Transaction(id) => DatasetHandleRole::Transaction(id),
    };
    if !operation.accepts_role(role) {
        return Err(
            build_runtime_error("dataset handle operation has a foreign sealed role")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let reference = owner.reference();
    let domain_facts =
        Digest::sha256(serde_json::to_vec(facts).expect("typed data handle input facts serialize"));
    select_host_effect(
        runtime,
        HostEffectRequest {
            operation: operation.public_name(),
            workload: HostEffectWorkload::DatasetHandle {
                operation,
                root: &owner.root,
                key: DatasetHandleKey {
                    host_identity: &reference.host_identity,
                    handle: reference.handle,
                    generation: reference.generation,
                },
                role,
                inputs: facts,
                // Every existing receiver binding creates a Value, including
                // logical acknowledgements when the caller discards them.
                output: HostEffectOutput::ReturnedValue,
            },
            domain_facts,
            filesystem: &owner.filesystem,
        },
    )
}

pub(super) async fn execute_facts<T>(
    runtime: Option<&RuntimeContext>,
    owner: &AuthorizedHandle,
    operation: DatasetHandleOperation,
    facts: &[DatasetHandleInput],
    effect: impl Future<Output = BuiltinResult<T>>,
) -> BuiltinResult<T> {
    let receipt = select_facts(runtime, owner, operation, facts)?;
    super::execution::execute(runtime, receipt, effect).await
}

pub(super) fn input(value: &Value) -> DatasetHandleInput {
    let scalar = match value {
        Value::String(text) => Some(ScalarControl::Text(text)),
        Value::CharArray(chars) => Some(ScalarControl::Character {
            shape: &chars.shape,
            data: &chars.data,
        }),
        Value::Bool(boolean) => Some(ScalarControl::Logical(*boolean)),
        Value::Int(integer) => Some(ScalarControl::Integer(integer)),
        Value::Num(number) => Some(ScalarControl::DoubleBits(number.to_bits())),
        Value::Complex(real, imaginary) => Some(ScalarControl::ComplexDoubleBits {
            real: real.to_bits(),
            imaginary: imaginary.to_bits(),
        }),
        _ => None,
    };
    let identity_digest = scalar
        .map(|scalar| {
            Digest::sha256(serde_json::to_vec(&scalar).expect("typed scalar control serializes"))
        })
        .or_else(|| native_numeric_identity(value));
    DatasetHandleInput {
        fact: crate::value_fact::value_fact(value),
        identity_digest,
    }
}

/// Borrow native payload through its exact typed accessor. This bounded
/// synchronous prefix is admission work, not a second materialized Value.
/// Larger arrays remain uncertified until the shared staged evidence route
/// can cooperatively digest their full payload before selection.
fn native_numeric_identity(value: &Value) -> Option<Digest> {
    const MAX_ADMISSION_ELEMENTS: usize = 4_096;
    if bounded_identity_work(value, 0)? > MAX_ADMISSION_ELEMENTS {
        return None;
    }
    let mut hasher = Sha256::new();
    match value {
        Value::Tensor(tensor) => {
            hash_piece(
                &mut hasher,
                &("real", &tensor.shape, tensor.numeric_dtype().class_name()),
            );
            for index in 0..tensor.len() {
                hash_piece(
                    &mut hasher,
                    &NumericAtom::from(tensor.numeric_value_at(index)?),
                );
            }
        }
        Value::ComplexTensor(tensor) => {
            hash_piece(
                &mut hasher,
                &(
                    "complex",
                    &tensor.shape,
                    tensor.numeric_dtype().class_name(),
                ),
            );
            for index in 0..tensor.len() {
                let (real, imaginary) = tensor.numeric_value_at(index)?;
                hash_piece(&mut hasher, &NumericAtom::from(real));
                hash_piece(&mut hasher, &NumericAtom::from(imaginary));
            }
        }
        Value::Cell(cell) => {
            hash_piece(&mut hasher, &("cell", &cell.shape));
            for child in &cell.data {
                hash_piece(&mut hasher, &input(child).identity_digest?);
            }
        }
        Value::Struct(structure) => {
            hash_piece(&mut hasher, &"struct");
            for (name, child) in &structure.fields {
                hash_piece(&mut hasher, name);
                hash_piece(&mut hasher, &input(child).identity_digest?);
            }
        }
        Value::LogicalArray(array) => {
            hash_piece(
                &mut hasher,
                &("logical", &array.shape, array.data.as_slice()),
            );
        }
        Value::StringArray(array) => {
            hash_piece(&mut hasher, &("strings", &array.shape, &array.data));
        }
        _ => return None,
    }
    Some(Digest::from_bytes(hasher.finalize().into()))
}

fn bounded_identity_work(value: &Value, depth: usize) -> Option<usize> {
    const MAX_ADMISSION_ELEMENTS: usize = 4_096;
    if depth > 64 {
        return None;
    }
    match value {
        Value::Tensor(tensor) => Some(tensor.len()),
        Value::ComplexTensor(tensor) => Some(tensor.len()),
        Value::LogicalArray(array) => Some(array.len()),
        Value::StringArray(array) => Some(array.data.len()),
        Value::Cell(cell) => cell.data.iter().try_fold(1usize, |total, child| {
            total
                .checked_add(bounded_identity_work(child, depth + 1)?)
                .filter(|total| *total <= MAX_ADMISSION_ELEMENTS)
        }),
        Value::Struct(structure) => structure.fields.values().try_fold(1usize, |total, child| {
            total
                .checked_add(bounded_identity_work(child, depth + 1)?)
                .filter(|total| *total <= MAX_ADMISSION_ELEMENTS)
        }),
        Value::String(_)
        | Value::CharArray(_)
        | Value::Bool(_)
        | Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _) => Some(1),
        _ => None,
    }
}

fn hash_piece(hasher: &mut Sha256, value: &impl serde::Serialize) {
    let bytes = serde_json::to_vec(value).expect("typed numeric identity serializes");
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(serde::Serialize)]
enum NumericAtom {
    F64(u64),
    F32(u32),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl From<NumericScalar> for NumericAtom {
    fn from(value: NumericScalar) -> Self {
        match value {
            NumericScalar::F64(value) => Self::F64(value.to_bits()),
            NumericScalar::F32(value) => Self::F32(value.to_bits()),
            NumericScalar::I8(value) => Self::I8(value),
            NumericScalar::I16(value) => Self::I16(value),
            NumericScalar::I32(value) => Self::I32(value),
            NumericScalar::I64(value) => Self::I64(value),
            NumericScalar::U8(value) => Self::U8(value),
            NumericScalar::U16(value) => Self::U16(value),
            NumericScalar::U32(value) => Self::U32(value),
            NumericScalar::U64(value) => Self::U64(value),
        }
    }
}

#[derive(serde::Serialize)]
enum ScalarControl<'a> {
    Text(&'a str),
    Character {
        shape: &'a [usize],
        data: &'a [char],
    },
    Logical(bool),
    Integer(&'a runmat_value::IntValue),
    DoubleBits(u64),
    ComplexDoubleBits {
        real: u64,
        imaginary: u64,
    },
}

#[derive(serde::Serialize)]
struct QueueIdentity {
    base_sequence: u64,
    revision: u64,
    writes: usize,
    resizes: usize,
    fills: usize,
    creates: usize,
    deletes: usize,
    attributes: usize,
}

pub(super) fn queue_input(tx: &crate::data::PendingTxn) -> DatasetHandleInput {
    let identity = QueueIdentity {
        base_sequence: tx.base_sequence,
        revision: tx.revision,
        writes: tx.writes.len(),
        resizes: tx.resizes.len(),
        fills: tx.fills.len(),
        creates: tx.create_arrays.len(),
        deletes: tx.delete_arrays.len(),
        attributes: tx.attrs.len(),
    };
    DatasetHandleInput {
        fact: runmat_types::ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        identity_digest: Some(Digest::sha256(
            serde_json::to_vec(&identity).expect("typed transaction queue identity serializes"),
        )),
    }
}

pub(super) async fn execute<T>(
    runtime: Option<&RuntimeContext>,
    owner: &AuthorizedHandle,
    operation: DatasetHandleOperation,
    arguments: &[&Value],
    effect: impl Future<Output = BuiltinResult<T>>,
) -> BuiltinResult<T> {
    let receipt = select(runtime, owner, operation, arguments)?;
    super::execution::execute(runtime, receipt, effect).await
}

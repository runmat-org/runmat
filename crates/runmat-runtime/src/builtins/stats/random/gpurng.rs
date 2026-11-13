//! Session-owned GPU random stream controls. Public commands never seed or
//! inspect the ordinary CPU RNG, and no process-global provider RNG hook is
//! used as language state authority.

use crate::builtins::common::random::gpu::{self, GpuRandomAlgorithm, GpuRandomSnapshot};
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;
use runmat_value::{IntegerStorage, Tensor, Value};
use std::sync::atomic::{AtomicU64, Ordering};

const NAME: &str = "gpurng";
const SEED_LIMIT: u64 = 1_u64 << 32;
const STATE_MAGIC: u32 = 0x524d_4731;
const STATE_WORDS: usize = 17;
static SHUFFLE_ORDINAL: AtomicU64 = AtomicU64::new(0);

#[runtime_builtin(
    name = "gpurng",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::gpurng"
)]
async fn gpurng_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let previous = gpu::snapshot()?;
    let Some(command) = parse_command(args.view())? else {
        return snapshot_to_value(previous);
    };
    let replacement = match command {
        Command::Default => GpuRandomSnapshot::default(),
        Command::Seed(seed) => GpuRandomSnapshot {
            seed,
            position: 0,
            ..previous
        },
        Command::Shuffle => GpuRandomSnapshot {
            seed: shuffle_seed(),
            position: 0,
            ..previous
        },
        Command::Restore(snapshot) => snapshot,
    };
    // Snapshot representation is prepared before mutating the session stream.
    let output = snapshot_to_value(previous)?;
    gpu::replace(replacement)?;
    Ok(output)
}

fn error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier("RunMat:gpurng:InvalidArgument")
        .build()
}

fn unsupported(name: &str) -> RuntimeError {
    build_runtime_error(format!("gpurng: generator '{name}' is not implemented"))
        .with_builtin(NAME)
        .with_identifier("RunMat:gpurng:GeneratorUnsupported")
        .build()
}

enum Command {
    Default,
    Seed(u64),
    Shuffle,
    Restore(GpuRandomSnapshot),
}

fn parse_command(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Option<Command>> {
    match (args.len(), args.get(0), args.get(1)) {
        (0, _, _) => Ok(None),
        (1, Some(value), _) => {
            if let Value::Struct(_) = value {
                return snapshot_from_value(value).map(Command::Restore).map(Some);
            }
            if let Some(keyword) = keyword_of(value)? {
                return match keyword.as_str() {
                    "default" => Ok(Some(Command::Default)),
                    "shuffle" => Ok(Some(Command::Shuffle)),
                    "threefry" => Ok(Some(Command::Seed(0))),
                    "philox" | "combrecursive" => Err(unsupported(&keyword)),
                    _ => Err(error(format!("gpurng: unknown option '{keyword}'"))),
                };
            }
            parse_seed(value).map(Command::Seed).map(Some)
        }
        (2, Some(seed), Some(generator)) => {
            let Some(generator) = keyword_of(generator)? else {
                return Err(error("gpurng: generator must be a string scalar"));
            };
            if generator != "threefry" {
                return Err(match generator.as_str() {
                    "philox" | "combrecursive" => unsupported(&generator),
                    _ => error(format!("gpurng: unknown generator '{generator}'")),
                });
            }
            parse_seed(seed).map(Command::Seed).map(Some)
        }
        _ => Err(error("gpurng: expected zero, one, or two arguments")),
    }
}

fn parse_seed(value: &Value) -> BuiltinResult<u64> {
    let seed = match value {
        Value::Int(value) => value.try_to_u64(),
        Value::Num(value) if value.is_finite() && *value >= 0.0 && value.fract() == 0.0 => {
            Some(*value as u64)
        }
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            if let Some(storage) = value.integer_storage() {
                storage.value_at(0).and_then(|value| value.try_to_u64())
            } else {
                let number = tensor::tensor_value_f64(value, 0);
                (number.is_finite() && number >= 0.0 && number.fract() == 0.0)
                    .then_some(number as u64)
            }
        }
        _ => None,
    }
    .ok_or_else(|| error("gpurng: seed must be a nonnegative integer"))?;
    (seed < SEED_LIMIT)
        .then_some(seed)
        .ok_or_else(|| error("gpurng: seed must be less than 2^32"))
}

fn shuffle_seed() -> u64 {
    let time = runmat_time::unix_timestamp_ns();
    let ordinal = SHUFFLE_ORDINAL.fetch_add(1, Ordering::Relaxed);
    ((time as u64) ^ ((time >> 32) as u64) ^ ordinal.rotate_left(19)) & (SEED_LIMIT - 1)
}

fn snapshot_to_value(snapshot: GpuRandomSnapshot) -> BuiltinResult<Value> {
    let mut words = vec![0_u32; STATE_WORDS];
    words[0] = STATE_MAGIC;
    words[1] = 1;
    words[2] = snapshot.seed as u32;
    words[3] = 1; // Threefry 4x64_20
    words[4] = 1; // BoxMuller
    for (index, slot) in words[5..9].iter_mut().enumerate() {
        *slot = (snapshot.position >> (index * 32)) as u32;
    }
    words[9] = words[..9]
        .iter()
        .fold(0x9e37_79b9_u32, |sum, word| sum ^ *word);
    let state = Tensor::new_integer(IntegerStorage::U32(words), vec![STATE_WORDS, 1])
        .map_err(|source| error(format!("gpurng: state construction failed: {source}")))?;
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    value
        .fields
        .insert("Type".to_owned(), Value::string("threefry".to_owned()));
    value
        .fields
        .insert("Seed".to_owned(), Value::Num(snapshot.seed as f64));
    value
        .fields
        .insert("State".to_owned(), Value::Tensor(state));
    Ok(Value::Struct(value.finish()))
}

fn snapshot_from_value(value: &Value) -> BuiltinResult<GpuRandomSnapshot> {
    let Value::Struct(value) = value else {
        return Err(error("gpurng: expected a prior state structure"));
    };
    if value.fields.len() != 3 {
        return Err(error(
            "gpurng: state structure must contain Type, Seed, and State",
        ));
    }
    let kind = value
        .fields
        .get("Type")
        .map(keyword_of)
        .transpose()?
        .flatten()
        .ok_or_else(|| error("gpurng: state Type must be a generator name"))?;
    if kind != "threefry" {
        return Err(unsupported(&kind));
    }
    let seed = parse_seed(
        value
            .fields
            .get("Seed")
            .ok_or_else(|| error("gpurng: state Seed is missing"))?,
    )?;
    let state = value
        .fields
        .get("State")
        .ok_or_else(|| error("gpurng: state State is missing"))?;
    let Value::Tensor(state) = state else {
        return Err(error("gpurng: State must be a uint32 column vector"));
    };
    if state.shape() != [STATE_WORDS, 1] {
        return Err(error("gpurng: State must contain 17 uint32 words"));
    }
    let Some(IntegerStorage::U32(words)) = state.integer_storage() else {
        return Err(error("gpurng: State must have native uint32 storage"));
    };
    if words.len() != STATE_WORDS
        || words[0] != STATE_MAGIC
        || words[1] != 1
        || words[2] != seed as u32
        || words[3] != 1
        || words[4] != 1
        || words[10..].iter().any(|word| *word != 0)
        || words[9]
            != words[..9]
                .iter()
                .fold(0x9e37_79b9_u32, |sum, word| sum ^ *word)
    {
        return Err(error("gpurng: State is not a valid RunMat GPU snapshot"));
    }
    let mut position = 0_u128;
    for (index, word) in words[5..9].iter().enumerate() {
        position |= u128::from(*word) << (index * 32);
    }
    Ok(GpuRandomSnapshot {
        algorithm: GpuRandomAlgorithm::Threefry4x64_20,
        seed,
        position,
        normal_transform: gpu::GpuNormalTransform::BoxMuller,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_query_and_restore_do_not_touch_cpu_stream() {
        let cpu_before = crate::builtins::common::random::snapshot().expect("CPU snapshot");
        gpu::replace(GpuRandomSnapshot::default()).expect("GPU reset");
        let original = futures::executor::block_on(gpurng_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("GPU query");
        gpu::reserve(7, NAME).expect("advance GPU stream");
        futures::executor::block_on(gpurng_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![original]),
        ))
        .expect("GPU restore");
        assert_eq!(gpu::snapshot().expect("restored").position, 0);
        let cpu_after = crate::builtins::common::random::snapshot().expect("CPU after");
        assert_eq!(cpu_after.state, cpu_before.state);
        assert_eq!(cpu_after.seed, cpu_before.seed);
        assert_eq!(cpu_after.algorithm(), cpu_before.algorithm());
    }

    #[test]
    fn unsupported_generator_does_not_mutate_the_gpu_child() {
        gpu::replace(GpuRandomSnapshot::default()).expect("GPU reset");
        let before = gpu::snapshot().expect("GPU snapshot");
        let error = futures::executor::block_on(gpurng_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("philox")]),
        ))
        .expect_err("unsupported generator must be rejected");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpurng:GeneratorUnsupported")
        );
        assert_eq!(gpu::snapshot().expect("GPU unchanged"), before);
    }
}

//! `rng` builtin for seeding and querying RunMat's host random generator.
#[cfg(test)]
use runmat_builtins::RNG_INTEGER_CAPABILITIES;
use runmat_builtins::{
    RNG_ERROR_CONTEXT_REQUIRED, RNG_ERROR_GENERATOR_UNSUPPORTED, RNG_ERROR_INTERNAL,
    RNG_ERROR_INVALID_ARGUMENT, RNG_ERROR_SEED_NONNEGATIVE, RNG_ERROR_STATE_TYPE_FIELD_MISSING,
    RNG_TYPED_STATE_EXTENSION, RNG_WIDE_SEED_EXTENSION,
};

use crate::builtins::common::random::{self, RngAlgorithm, RngSnapshot, DEFAULT_USER_SEED};
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, IntegerStorage, Tensor, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rng";
const MATLAB_SEED_UPPER_BOUND: u64 = 1_u64 << 32;

fn rng_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rng_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rng_error_with(error, error.message)
}

fn rng_internal_error(message: impl Into<String>) -> RuntimeError {
    rng_error_with(&RNG_ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "rng",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::random::rng"
)]
async fn rng_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        let mut turn = random::acquire_turn(runtime.as_ref(), NAME).await?;
        let current = random::snapshot_in_turn(&mut turn)?;
        return snapshot_to_value(current);
    }

    ensure_rng_state_extensions(args.view())?;
    let command = parse_command(args.view())?;
    let mut turn = random::acquire_turn(runtime.as_ref(), NAME).await?;
    let previous = random::snapshot_in_turn(&mut turn)?;
    apply_command(runtime.as_ref(), &mut turn, command)?;
    // The explicit GPU stream is independent. A provider's Philox state is
    // not the host MT19937 state and must not be relabeled or reseeded here.
    snapshot_to_value(previous)
}

#[derive(Debug, Clone)]
enum ParsedCommand {
    Default,
    Seed(u64),
    Shuffle,
    Restore(RngSnapshot),
}

fn parse_command(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<ParsedCommand> {
    match args.len() {
        1 => parse_single_arg(&args[0]),
        2 => parse_double_args(&args[0], &args[1]),
        _ => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: invalid number of arguments",
        )),
    }
}

fn parse_single_arg(arg: &Value) -> BuiltinResult<ParsedCommand> {
    if let Some(keyword) = keyword_of(arg)? {
        return parse_keyword(&keyword, None);
    }
    match arg {
        Value::Struct(_) => Ok(ParsedCommand::Restore(snapshot_from_value(arg)?)),
        _ => {
            let seed = parse_seed_scalar(arg, "rng: seed")?;
            ensure_direct_seed_range(seed)?;
            Ok(ParsedCommand::Seed(seed))
        }
    }
}

fn parse_double_args(first: &Value, second: &Value) -> BuiltinResult<ParsedCommand> {
    if let Some(keyword) = keyword_of(first)? {
        let generator = Some(parse_generator(second)?);
        return parse_keyword(&keyword, generator);
    }
    let seed = parse_seed_scalar(first, "rng: seed")?;
    ensure_direct_seed_range(seed)?;
    let generator = parse_generator(second)?;
    if generator != RngAlgorithm::Twister {
        return Err(rng_error_with(
            &RNG_ERROR_GENERATOR_UNSUPPORTED,
            format!(
                "rng: generator '{}' is not supported for new seeds",
                generator.as_str()
            ),
        ));
    }
    Ok(ParsedCommand::Seed(seed))
}

fn parse_keyword(keyword: &str, generator: Option<RngAlgorithm>) -> BuiltinResult<ParsedCommand> {
    let algo = generator.unwrap_or(RngAlgorithm::Twister);
    if algo != RngAlgorithm::Twister {
        return Err(rng_error_with(
            &RNG_ERROR_GENERATOR_UNSUPPORTED,
            format!(
                "rng: generator '{}' is not supported in RunMat",
                algo.as_str()
            ),
        ));
    }
    match keyword {
        "default" | "twister" => Ok(ParsedCommand::Default),
        "shuffle" => Ok(ParsedCommand::Shuffle),
        other => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("rng: unknown option '{other}'"),
        )),
    }
}

fn apply_command(
    runtime: Option<&crate::context::RuntimeContext>,
    turn: &mut random::RandomTurn,
    command: ParsedCommand,
) -> BuiltinResult<()> {
    match command {
        ParsedCommand::Default => {
            random::set_default_in_turn(turn)?;
            Ok(())
        }
        ParsedCommand::Seed(seed) => {
            random::set_seed_in_turn(turn, seed)?;
            Ok(())
        }
        ParsedCommand::Shuffle => {
            let seed = shuffle_seed(runtime)?;
            random::set_seed_in_turn(turn, seed)?;
            Ok(())
        }
        ParsedCommand::Restore(snapshot) => {
            random::apply_snapshot_in_turn(turn, snapshot)?;
            Ok(())
        }
    }
}

mod portable;
pub(crate) use portable::PortableSnapshot;

pub(crate) fn snapshot_to_value(snapshot: RngSnapshot) -> BuiltinResult<Value> {
    PortableSnapshot::capture(&snapshot).public_value()
}

pub(crate) fn snapshot_from_value(value: &Value) -> BuiltinResult<RngSnapshot> {
    let Value::Struct(struct_value) = value else {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: expected a structure with fields Type, Seed, and State",
        ));
    };
    let type_value = struct_value
        .fields
        .get("Type")
        .or_else(|| struct_value.fields.get("type"))
        .ok_or_else(|| rng_error(&RNG_ERROR_STATE_TYPE_FIELD_MISSING))?;
    let generator = match keyword_of(type_value)? {
        Some(ref kw) => parse_generator_keyword(kw)?,
        None => {
            return Err(rng_error_with(
                &RNG_ERROR_INVALID_ARGUMENT,
                "rng: Type field must be a string",
            ))
        }
    };

    let seed_opt = struct_value
        .fields
        .get("Seed")
        .or_else(|| struct_value.fields.get("seed"))
        .map(|v| parse_seed_scalar(v, "rng: Seed"))
        .transpose()?;
    let state_value = struct_value
        .fields
        .get("State")
        .or_else(|| struct_value.fields.get("state"))
        .ok_or_else(|| {
            rng_error_with(
                &RNG_ERROR_INVALID_ARGUMENT,
                "rng: state struct is missing the 'State' field",
            )
        })?;
    let state = match generator {
        RngAlgorithm::Twister => parse_mt_state(state_value)?,
        RngAlgorithm::RunMatLcgV1 => random::RandomState::Lcg(parse_state_scalar(state_value)?),
    };
    Ok(RngSnapshot {
        state,
        seed: seed_opt,
    })
}

fn parse_generator(value: &Value) -> BuiltinResult<RngAlgorithm> {
    match keyword_of(value)? {
        Some(keyword) => parse_generator_keyword(&keyword),
        None => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: generator name must be a string",
        )),
    }
}

fn parse_generator_keyword(keyword: &str) -> BuiltinResult<RngAlgorithm> {
    match keyword {
        "twister" | "default" => Ok(RngAlgorithm::Twister),
        "runmat-lcg-v1" => Ok(RngAlgorithm::RunMatLcgV1),
        other => Err(rng_error_with(
            &RNG_ERROR_GENERATOR_UNSUPPORTED,
            format!("rng: generator '{other}' is not supported"),
        )),
    }
}

fn parse_seed_scalar(value: &Value, label: &str) -> BuiltinResult<u64> {
    match value {
        Value::Int(i) => i.try_to_u64().ok_or_else(|| {
            rng_error_with(
                &RNG_ERROR_SEED_NONNEGATIVE,
                format!("{label}: seed must be non-negative"),
            )
        }),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    format!("{label}: seed must be finite"),
                ));
            }
            if *n < 0.0 {
                return Err(rng_error_with(
                    &RNG_ERROR_SEED_NONNEGATIVE,
                    format!("{label}: seed must be non-negative"),
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    format!("{label}: seed must be an integer"),
                ));
            }
            if rounded > (1u64 << 53) as f64 {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    format!("{label}: seed exceeds 53-bit integer precision"),
                ));
            }
            Ok(rounded as u64)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                int.try_to_u64().ok_or_else(|| {
                    rng_error_with(
                        &RNG_ERROR_SEED_NONNEGATIVE,
                        format!("{label}: seed must be non-negative"),
                    )
                })
            } else {
                parse_seed_scalar(&Value::Num(tensor::tensor_value_f64(t, 0)), label)
            }
        }
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: expected a numeric seed"),
        )),
        _ => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: expected a scalar numeric seed"),
        )),
    }
}

fn ensure_direct_seed_range(seed: u64) -> BuiltinResult<()> {
    if seed >= MATLAB_SEED_UPPER_BOUND {
        crate::compatibility::ensure_builtin_extension_enabled(&RNG_WIDE_SEED_EXTENSION, NAME)?;
    }
    Ok(())
}

fn ensure_rng_state_extensions(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    if args.len() == 1 {
        if let Some(value) = args.first() {
            ensure_snapshot_extension(value, crate::context::legacy::active().as_ref())?;
        }
    }
    Ok(())
}
pub(crate) fn restore_snapshot_value(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<RngSnapshot> {
    ensure_snapshot_extension(value, runtime)?;
    snapshot_from_value(value)
}
fn ensure_snapshot_extension(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    let Value::Struct(state) = value else {
        return Ok(());
    };
    let legacy_state = state
        .fields
        .get("Type")
        .or_else(|| state.fields.get("type"))
        .map(keyword_of)
        .transpose()?
        .flatten()
        .as_deref()
        == Some("runmat-lcg-v1");
    let fields: &[&str] = if legacy_state {
        &["Seed", "seed", "State", "state"]
    } else {
        &["Seed", "seed"]
    };
    if fields
        .iter()
        .copied()
        .filter_map(|name| state.fields.get(name))
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            &RNG_TYPED_STATE_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn parse_mt_state(value: &Value) -> BuiltinResult<random::RandomState> {
    let Value::Tensor(tensor) = value else {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: twister State must be a 625-by-1 uint32 array",
        ));
    };
    if tensor.shape() != [625, 1] {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: twister State must be a 625-by-1 uint32 array",
        ));
    }
    let Some(IntegerStorage::U32(values)) = tensor.integer_storage() else {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: twister State must have uint32 storage",
        ));
    };
    let words: [u32; 624] = values[..624].try_into().map_err(|_| {
        rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: invalid twister state length",
        )
    })?;
    random::RandomState::from_mt_words(words, values[624] as usize).map_err(|error| {
        rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("rng: invalid twister state: {error}"),
        )
    })?
}

fn parse_state_scalar(value: &Value) -> BuiltinResult<u64> {
    match value {
        Value::Tensor(t) => match tensor::element_count(t.shape()) {
            1 => {
                if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                    int.try_to_u64().ok_or_else(|| {
                        rng_error_with(
                            &RNG_ERROR_INVALID_ARGUMENT,
                            "rng: State must be non-negative",
                        )
                    })
                } else {
                    parse_state_scalar(&Value::Num(tensor::tensor_value_f64(t, 0)))
                }
            }
            2 => {
                let (lo, hi) = if let Some(storage) = t.integer_storage() {
                    let lo = storage.value_at(0).ok_or_else(|| {
                        rng_error_with(
                            &RNG_ERROR_INVALID_ARGUMENT,
                            "rng: State tensor must contain one or two elements",
                        )
                    })?;
                    let hi = storage.value_at(1).ok_or_else(|| {
                        rng_error_with(
                            &RNG_ERROR_INVALID_ARGUMENT,
                            "rng: State tensor must contain one or two elements",
                        )
                    })?;
                    (
                        parse_state_word_int(lo, "rng: State[1]")?,
                        parse_state_word_int(hi, "rng: State[2]")?,
                    )
                } else {
                    (
                        parse_state_word(tensor::tensor_value_f64(t, 0), "rng: State[1]")?,
                        parse_state_word(tensor::tensor_value_f64(t, 1), "rng: State[2]")?,
                    )
                };
                Ok(lo | (hi << 32))
            }
            _ => Err(rng_error_with(
                &RNG_ERROR_INVALID_ARGUMENT,
                "rng: State tensor must contain one or two elements",
            )),
        },
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    "rng: State must be finite",
                ));
            }
            if *n < 0.0 {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    "rng: State must be non-negative",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    "rng: State must be an integer vector",
                ));
            }
            Ok(rounded as u64)
        }
        Value::Int(i) => i.try_to_u64().ok_or_else(|| {
            rng_error_with(
                &RNG_ERROR_INVALID_ARGUMENT,
                "rng: State must be non-negative",
            )
        }),
        other => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("rng: unsupported State value {other:?}"),
        )),
    }
}

fn parse_state_word_int(value: IntValue, label: &str) -> BuiltinResult<u64> {
    let word = value.try_to_u64().ok_or_else(|| {
        rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must be non-negative"),
        )
    })?;
    if word > u32::MAX as u64 {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must fit in uint32"),
        ));
    }
    Ok(word)
}

fn parse_state_word(value: f64, label: &str) -> BuiltinResult<u64> {
    if !value.is_finite() {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must be finite"),
        ));
    }
    if value < 0.0 {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must be non-negative"),
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must be an integer"),
        ));
    }
    if rounded > (u32::MAX as f64) {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: exceeds uint32 precision"),
        ));
    }
    Ok(rounded as u64)
}

fn shuffle_seed(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<u64> {
    let runtime = runtime.ok_or_else(|| {
        let mut error = rng_error(&RNG_ERROR_CONTEXT_REQUIRED);
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })?;
    runtime
        .service_ports()
        .require_selected_local_effect(runmat_types::EffectKind::EnvironmentRead)?;
    if let Ok(env) = runtime
        .environment_state()
        .borrow()
        .var("RUNMAT_RNG_SHUFFLE_SEED")
    {
        if let Ok(parsed) = env.parse::<u64>() {
            return Ok(parsed);
        }
    }
    let now = runtime
        .service_ports()
        .timing()
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?
        .wall_time_since_epoch()
        .as_nanos();
    let mut seed = now as u64 ^ (now >> 32) as u64;
    // Stable folding of the admitted clock value, without host entropy or
    // address salts: a deterministic Clock must produce a deterministic seed.
    seed ^= seed << 7;
    Ok(if seed == 0 {
        DEFAULT_USER_SEED.wrapping_add(1)
    } else {
        seed
    })
}

#[cfg(test)]
#[path = "rng/shuffle_tests.rs"]
mod shuffle_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};
    use futures::executor::block_on;
    use runmat_builtins::BuiltinIntegerInputAvailability;
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn draw_uniform(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
        let mut turn = random::try_acquire_turn(None, label)?;
        random::generate_uniform(&mut turn, len, label)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_returns_current_state() {
        let _guard = random::test_guard();
        random::reset_rng();
        let value = block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("rng");
        let snapshot = snapshot_from_value(&value).expect("snapshot");
        assert_eq!(snapshot.state, random::default_snapshot().state);
        assert_eq!(snapshot.seed, Some(DEFAULT_USER_SEED));
        assert_eq!(snapshot.algorithm(), RngAlgorithm::Twister);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_seed_is_reproducible() {
        let _guard = random::test_guard();
        random::reset_rng();
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U32(42),
            )]),
        ))
        .expect("rng");
        let seq1 = draw_uniform(5, "rng test").expect("uniform");
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U32(42),
            )]),
        ))
        .expect("rng");
        let seq2 = draw_uniform(5, "rng test").expect("uniform");
        assert_eq!(seq1, seq2);
    }

    #[test]
    fn rng_integer_seed_and_state_preserve_full_uint64_range() {
        assert_eq!(
            parse_seed_scalar(&Value::Int(IntValue::U64(u64::MAX)), "rng: seed")
                .expect("uint64 seed"),
            u64::MAX
        );
        assert_eq!(
            parse_state_scalar(&Value::Int(IntValue::U64(u64::MAX))).expect("uint64 state"),
            u64::MAX
        );
        assert!(parse_seed_scalar(&Value::Int(IntValue::I64(-1)), "rng: seed").is_err());
    }

    #[test]
    fn rng_wide_seed_is_gated_and_query_preserves_exact_seed() {
        let _guard = random::test_guard();
        random::reset_rng();
        let wide = (1_u64 << 53) + 1;
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(rng_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                    IntValue::U64(wide),
                )]),
            ))
            .expect_err("wide seed must gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:RngWideSeedExtension")
            );
        }
        {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            block_on(rng_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                    IntValue::U64(wide),
                )]),
            ))
            .expect("RunMat wide seed");
            let Value::Struct(snapshot) = block_on(rng_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("snapshot") else {
                panic!("expected snapshot struct");
            };
            assert_eq!(
                snapshot.fields.get("Seed"),
                Some(&Value::Int(IntValue::U64(wide)))
            );
        }
    }

    #[test]
    fn rng_typed_state_fields_are_an_independent_extension() {
        let Value::Struct(mut state) =
            snapshot_to_value(random::default_snapshot()).expect("state snapshot")
        else {
            panic!("expected state struct");
        };
        crate::builtins::common::test_support::replace_struct_field_unbounded(
            &mut state,
            "Seed",
            Value::Int(IntValue::U32(7)),
        );
        let value = || Value::Struct(state.clone());
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(rng_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![value()]),
            ))
            .expect_err("typed state must gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:RngTypedStateFieldsExtension")
            );
        }
        {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            block_on(rng_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![value()]),
            ))
            .expect("typed state extension");
        }
    }

    #[test]
    fn rng_integer_capabilities_cover_public_and_extended_seed_domains() {
        assert_eq!(RNG_INTEGER_CAPABILITIES.len(), 4);
        assert_eq!(
            RNG_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            RNG_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(
            RNG_INTEGER_CAPABILITIES[2].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            RNG_INTEGER_CAPABILITIES[3].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
    }

    #[test]
    fn rng_typed_integer_tensor_seed_and_state_are_exact() {
        let seed =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("seed");
        assert_eq!(
            parse_seed_scalar(&Value::Tensor(seed), "rng: seed").expect("typed seed"),
            u64::MAX
        );

        let scalar_state =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("state");
        assert_eq!(
            parse_state_scalar(&Value::Tensor(scalar_state)).expect("typed state"),
            u64::MAX
        );

        let word_state = Tensor::new_integer(
            IntegerStorage::U64(vec![u32::MAX as u64, u32::MAX as u64]),
            vec![1, 2],
        )
        .expect("state words");
        assert_eq!(
            parse_state_scalar(&Value::Tensor(word_state)).expect("typed state words"),
            u64::MAX
        );

        let negative_seed =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("seed");
        assert!(parse_seed_scalar(&Value::Tensor(negative_seed), "rng: seed").is_err());

        let wide_word = Tensor::new_integer(
            IntegerStorage::U64(vec![u32::MAX as u64 + 1, 0]),
            vec![1, 2],
        )
        .expect("state word");
        assert!(parse_state_scalar(&Value::Tensor(wide_word)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn rng_restores_legacy_lcg_rank_zero_and_rank_one_state_tensors() {
        let _guard = random::test_guard();
        random::reset_rng();

        for (state, expected) in [
            (Tensor::new(vec![7.0], Vec::new()).unwrap(), 7_u64),
            (
                Tensor::new(vec![3.0, 4.0], vec![2]).unwrap(),
                3_u64 | (4_u64 << 32),
            ),
        ] {
            let Value::Struct(mut snapshot) = block_on(rng_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("rng snapshot") else {
                panic!("rng snapshot struct");
            };
            crate::builtins::common::test_support::replace_struct_field_unbounded(
                &mut snapshot,
                "State",
                Value::Tensor(state),
            );
            crate::builtins::common::test_support::replace_struct_field_unbounded(
                &mut snapshot,
                "Type",
                Value::String(RngAlgorithm::RunMatLcgV1.as_str().to_string().into()),
            );
            block_on(rng_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
                    snapshot,
                )]),
            ))
            .expect("restore state");
            assert_eq!(
                random::snapshot().unwrap().state,
                random::RandomState::Lcg(expected)
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_restore_struct_roundtrip() {
        let _guard = random::test_guard();
        random::reset_rng();
        let saved = block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("rng");
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U32(7),
            )]),
        ))
        .expect("rng");
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![saved.clone()]),
        ))
        .expect("rng restore");
        let current = random::snapshot().expect("snapshot");
        assert_eq!(current.state, random::default_snapshot().state);
        assert_eq!(current.seed, Some(DEFAULT_USER_SEED));
    }

    #[test]
    fn rng_twister_state_has_runmat_625_word_wire_and_restores_midstream() {
        let _guard = random::test_guard();
        random::reset_rng();
        random::set_seed(41).expect("seed");
        let _ = draw_uniform(319, "rng test").expect("advance across twist");
        let saved = block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("query");
        let Value::Struct(fields) = &saved else {
            panic!("expected state struct");
        };
        let Value::Tensor(state) = fields.fields.get("State").expect("State") else {
            panic!("expected state tensor");
        };
        assert_eq!(state.shape(), &[625, 1]);
        let Some(IntegerStorage::U32(words)) = state.integer_storage() else {
            panic!("expected uint32 state");
        };
        assert_eq!(words[624], 14);
        let expected = draw_uniform(8, "rng test").expect("continuation");
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![saved]),
        ))
        .expect("restore");
        assert_eq!(draw_uniform(8, "rng test").expect("replay"), expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_default_restores_state() {
        let _guard = random::test_guard();
        random::reset_rng();
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U32(99),
            )]),
        ))
        .expect("seed rng");
        let previous = block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("default")]),
        ))
        .expect("rng default");
        let restored = random::snapshot().expect("snapshot");
        assert_eq!(restored.state, random::default_snapshot().state);
        assert_eq!(restored.seed, Some(DEFAULT_USER_SEED));
        let prev_snapshot = snapshot_from_value(&previous).expect("prev snapshot");
        assert_eq!(prev_snapshot.seed, Some(99));
        assert_ne!(prev_snapshot.state, restored.state);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_seed_with_twister_alias() {
        let _guard = random::test_guard();
        random::reset_rng();
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U32(123),
            )]),
        ))
        .expect("rng seed first");
        let host_seq = draw_uniform(4, "twister alias host").expect("uniform");
        random::reset_rng();
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::U32(123)),
                Value::from("twister"),
            ]),
        ))
        .expect("rng seed twister");
        let alias_seq = draw_uniform(4, "twister alias verify").expect("uniform");
        assert_eq!(host_seq, alias_seq);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_rejects_negative_seed() {
        let _guard = random::test_guard();
        random::reset_rng();
        let err = block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::I32(-5),
            )]),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), RNG_ERROR_SEED_NONNEGATIVE.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_rejects_unknown_generator() {
        let _guard = random::test_guard();
        random::reset_rng();
        let err = block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("default"),
                Value::from("philox"),
            ]),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), RNG_ERROR_GENERATOR_UNSUPPORTED.identifier);
        let legacy = block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::U32(41)),
                Value::from("runmat-lcg-v1"),
            ]),
        ))
        .expect_err("legacy replay type must not silently select MT for a new seed");
        assert_eq!(
            legacy.identifier(),
            RNG_ERROR_GENERATOR_UNSUPPORTED.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_state_struct_requires_type() {
        let _guard = random::test_guard();
        random::reset_rng();
        let tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).expect("tensor");
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields.insert("Seed".to_string(), Value::Num(0.0));
        st.fields.insert("State".to_string(), Value::Tensor(tensor));
        let err = block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
                st.finish(),
            )]),
        ))
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            RNG_ERROR_STATE_TYPE_FIELD_MISSING.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_does_not_consume_or_reseed_explicit_provider_stream() {
        let _guard = random::test_guard();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            block_on(rng_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                    IntValue::U32(9),
                )]),
            ))
            .expect("rng");
            let before_provider = random::snapshot().expect("host snapshot");
            let handle = provider
                .random_uniform(&[4, 1])
                .expect("explicit gpu uniform");
            let _gpu = test_support::gather(Value::GpuTensor(handle.clone())).expect("gather");
            assert_eq!(random::snapshot().expect("host snapshot"), before_provider);
            drop(handle);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_seed_does_not_reseed_the_independent_gpu_child() {
        let _guard = random::test_guard();
        random::reset_rng();
        let gpu = crate::builtins::common::random::gpu::GpuRandomSnapshot::default();
        crate::builtins::common::random::gpu::replace(gpu).expect("GPU default");
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U32(9),
            )]),
        ))
        .expect("CPU seed");
        assert_eq!(
            crate::builtins::common::random::gpu::snapshot().expect("GPU snapshot"),
            gpu
        );
        let cpu_after_seed = random::snapshot().expect("CPU snapshot");
        crate::builtins::common::random::gpu::reserve(4, "test GPU child")
            .expect("GPU child range");
        let cpu_after_gpu = random::snapshot().expect("CPU unchanged");
        assert_eq!(cpu_after_gpu.state, cpu_after_seed.state);
        assert_eq!(cpu_after_gpu.seed, cpu_after_seed.seed);
        assert_eq!(cpu_after_gpu.algorithm(), cpu_after_seed.algorithm());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn legacy_wgpu_uniform_values_stay_in_range() {
        let _guard = random::test_guard();
        random::reset_rng();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        block_on(rng_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                IntValue::U32(2024),
            )]),
        ))
        .expect("rng wgpu seed");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let handle = provider
            .random_uniform(&[1, 6])
            .expect("wgpu random uniform");
        let gpu = test_support::gather(Value::GpuTensor(handle.clone())).expect("wgpu gather");
        let host = draw_uniform(6, "rng wgpu parity").expect("host uniform sequence");
        assert_eq!(gpu.len(), host.len());
        for (idx, value) in gpu.materialize_f64().iter().enumerate() {
            assert!(value.is_finite(), "gpu value at {idx} not finite: {value}");
            assert!(
                *value >= 0.0 && *value < 1.0,
                "gpu value at {idx} out of [0,1): {value}"
            );
        }
        for (idx, value) in host.iter().enumerate() {
            assert!(value.is_finite(), "host value at {idx} not finite: {value}");
            assert!(
                *value >= 0.0 && *value < 1.0,
                "host value at {idx} out of [0,1): {value}"
            );
        }
        drop(handle);
    }
}

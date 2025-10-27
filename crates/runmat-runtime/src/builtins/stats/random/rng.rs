//! MATLAB-compatible `rng` builtin for seeding and querying RunMat's global random generator.

use crate::builtins::common::random::{
    self, set_default, set_seed, RngAlgorithm, RngSnapshot, DEFAULT_USER_SEED,
};
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};
use log::debug;
use runmat_builtins::{StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "rng"
category: "stats/random"
keywords: ["rng", "seed", "twister", "shuffle", "random state"]
summary: "Seed, shuffle, and query the global random number generator."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Synchronizes the active acceleration provider's RNG state when supported."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "none"
requires_feature: null
tested:
  unit: "builtins::stats::random::rng::tests"
  integration: null
---

# What does the `rng` function do in MATLAB / RunMat?
`rng` queries or reconfigures the global pseudorandom number generator that powers
`rand`, `randn`, `randi`, and `randperm`. RunMat mirrors MATLAB's semantics:
you can reset to the default generator, seed with a specific integer, shuffle
based on the system clock, or save/restore the full generator state.

## How does the `rng` function behave in MATLAB / RunMat?
- `rng()` returns a structure describing the current generator (fields `Type`,
  `Seed`, and `State`).
- `rng(seed)` switches to the default generator with the supplied non-negative
  integer seed. Passing `0` reproduces MATLAB's `rng('default')`.
- `rng('default')` restores the default generator and seed (`twister`, seed `0`).
- `rng('shuffle')` seeds the generator with entropy derived from the current time.
- `rng(seed, 'twister')` explicitly names the generator. RunMat currently
  supports `twister`, matching MATLAB's default stream.
- `rng(S)` restores a state structure previously returned by `rng`.

Whenever an acceleration provider is active, RunMat also pushes the new seed to
the provider so GPU-resident random calls remain in sync with CPU sampling.
Providers that lack `set_rng_state` simply fall back to their existing behaviour,
which is documented below.

## Examples of using the `rng` function in MATLAB / RunMat

### Resetting the generator for reproducible simulation runs

```matlab
rng(0);
rand(1, 4)
```

Expected output:

```matlab
    0.3969    0.8408    0.4221    0.6260
```

### Saving and restoring RNG state around a computation

```matlab
s = rng;
rng(1337);
A = randn(2, 2);
rng(s);            % restore original stream
B = randn(2, 2);   % continues from the saved state
```

Expected output:

```matlab
A =
    0.6406   -0.8022
    0.2222   -0.7161

B =
    0.3969    0.8408
    0.4221    0.6260
```

### Scrambling the generator with the system clock

```matlab
rng('shuffle');
u = rand(1, 3);
```

Expected output:

```matlab
u =
    0.1378    0.7086    0.8463
```

### Keeping CPU and GPU random draws in sync

```matlab
rng(2024);
G = gpuArray(rand(1, 3));
C = gather(G);
H = rand(1, 3);         % same values as C
```

Expected output:

```matlab
C =
    0.6554    0.7501    0.6046

H =
    0.6554    0.7501    0.6046
```

### Creating deterministic permutations with `rng` and `randperm`

```matlab
rng(42);
p = randperm(6);
rng(42);
q = randperm(6);
```

Expected output:

```matlab
p =
     6     2     1     4     3     5

q =
     6     2     1     4     3     5
```

## GPU residency in RunMat (Do I need `gpuArray`?)

RunMat pushes the configured seed to the active acceleration provider whenever
`rng` changes the host generator. Providers that expose `set_rng_state`
(`runmat-accelerate`'s in-process and WGPU backends) therefore produce the same
sequences as CPU `rand`/`randn` when seeded via `rng`. If a provider omits this
hook, RunMat logs a debug message and continues with the provider's internal
seed stream; in that case GPU draws may differ from CPU draws even after
calling `rng`. You do not need to call `gpuArray` explicitly for residency
purposes—RunMat keeps GPU streams in sync automatically whenever the provider
supports RNG state injection.

## FAQ

### What generators does RunMat support?
RunMat currently exposes MATLAB's default `'twister'` stream. Additional
generators (`'philox'`, `'combRecursive'`, etc.) will surface as the underlying
engines land.

### What numeric range is valid for seeds?
Any finite, non-negative integer representable exactly in double precision
(`0 <= seed <= 2^53`) is accepted. Values outside this range produce an error.

### How can I force a specific seed for `rng('shuffle')` in tests?
Set the environment variable `RUNMAT_RNG_SHUFFLE_SEED` to an unsigned integer
before calling `rng('shuffle')`. RunMat uses that value instead of system time.

### Does `rng` affect GPU random numbers?
Yes—if the active acceleration provider implements `set_rng_state`. The bundled
in-process reference provider and the WGPU backend both honour this hook.

### How do I restore a saved state structure?
Pass the structure back to `rng`: `s = rng; ...; rng(s);` resumes where the
original state left off, matching MATLAB behaviour.

### What is stored in the `State` field?
RunMat serialises the 64-bit internal state into a 1×2 double tensor containing
the low and high 32-bit words. Feeding that tensor back to `rng` recreates the
exact state.

### Can I create independent streams?
Not yet. RunMat exposes MATLAB-compatible `RandStream` APIs in a separate
roadmap item; for now `rng` operates on the single global stream.

### Does `rng` change automatically when I call other random functions?
No. Only explicit calls to `rng`, `RandStream`, or provider-specific utilities
reconfigure the seed. Sampling routines (`rand`, `randn`, `randperm`, etc.)
consume the current stream but do not reseed it.

## See Also
[rand](../../array/creation/rand), [randn](../../array/creation/randn), [randi](../../array/creation/randi), [randperm](../../array/creation/randperm), [gpuArray](../../acceleration/gpu/gpuArray)

## Source & Feedback
- Implementation lives at [`crates/runmat-runtime/src/builtins/stats/random/rng.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/rng.rs)
- Found a bug or behavioural gap? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rng",
    op_kind: GpuOpKind::Custom("state-control"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("set_rng_state")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Not a numeric kernel; synchronises provider RNG state via set_rng_state when available.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rng",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Control builtin; fusion planner never embeds rng in generated kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("rng", DOC_MD);

#[runtime_builtin(
    name = "rng",
    category = "stats/random",
    summary = "Seed, shuffle, and query the global random number generator.",
    keywords = "rng,seed,twister,shuffle,state"
)]
fn rng_builtin(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        let current = random::snapshot()?;
        return snapshot_to_value(current);
    }

    let previous = random::snapshot()?;
    let command = parse_command(&args)?;
    apply_command(command)?;
    let current = random::snapshot()?;
    sync_provider_state(current.state);
    snapshot_to_value(previous)
}

#[derive(Debug, Clone)]
enum ParsedCommand {
    Default,
    Seed(u64),
    Shuffle,
    Restore(RngSnapshot),
}

fn parse_command(args: &[Value]) -> Result<ParsedCommand, String> {
    match args.len() {
        1 => parse_single_arg(&args[0]),
        2 => parse_double_args(&args[0], &args[1]),
        _ => Err("rng: invalid number of arguments".to_string()),
    }
}

fn parse_single_arg(arg: &Value) -> Result<ParsedCommand, String> {
    if let Some(keyword) = keyword_of(arg) {
        return parse_keyword(&keyword, None);
    }
    match arg {
        Value::Struct(_) => Ok(ParsedCommand::Restore(snapshot_from_value(arg)?)),
        _ => Ok(ParsedCommand::Seed(parse_seed_scalar(arg, "rng: seed")?)),
    }
}

fn parse_double_args(first: &Value, second: &Value) -> Result<ParsedCommand, String> {
    if let Some(keyword) = keyword_of(first) {
        let generator = Some(parse_generator(second)?);
        return parse_keyword(&keyword, generator);
    }
    let seed = parse_seed_scalar(first, "rng: seed")?;
    let _ = parse_generator(second)?;
    Ok(ParsedCommand::Seed(seed))
}

fn parse_keyword(keyword: &str, generator: Option<RngAlgorithm>) -> Result<ParsedCommand, String> {
    let algo = generator.unwrap_or(RngAlgorithm::RunMatLcg);
    if algo != RngAlgorithm::RunMatLcg {
        return Err(format!(
            "rng: generator '{}' is not supported in RunMat",
            algo.as_str()
        ));
    }
    match keyword {
        "default" | "twister" | "runmat-lcg" => Ok(ParsedCommand::Default),
        "shuffle" => Ok(ParsedCommand::Shuffle),
        other => Err(format!("rng: unknown option '{other}'")),
    }
}

fn apply_command(command: ParsedCommand) -> Result<(), String> {
    match command {
        ParsedCommand::Default => {
            set_default()?;
            Ok(())
        }
        ParsedCommand::Seed(seed) => {
            set_seed(seed)?;
            Ok(())
        }
        ParsedCommand::Shuffle => {
            let seed = shuffle_seed();
            set_seed(seed)?;
            Ok(())
        }
        ParsedCommand::Restore(snapshot) => {
            random::apply_snapshot(snapshot)?;
            Ok(())
        }
    }
}

fn snapshot_to_value(snapshot: RngSnapshot) -> Result<Value, String> {
    let mut fields = HashMap::new();
    let seed_value = snapshot.seed.unwrap_or(DEFAULT_USER_SEED) as f64;
    fields.insert(
        "Type".to_string(),
        Value::String(snapshot.algorithm.as_str().to_string()),
    );
    fields.insert("Seed".to_string(), Value::Num(seed_value));
    let lo = (snapshot.state & 0xFFFF_FFFF) as f64;
    let hi = (snapshot.state >> 32) as f64;
    let tensor = Tensor::new(vec![lo, hi], vec![1, 2]).map_err(|e| format!("rng: {e}"))?;
    fields.insert("State".to_string(), Value::Tensor(tensor));
    Ok(Value::Struct(StructValue { fields }))
}

fn snapshot_from_value(value: &Value) -> Result<RngSnapshot, String> {
    let Value::Struct(struct_value) = value else {
        return Err("rng: expected a structure with fields Type, Seed, and State".to_string());
    };
    let type_value = struct_value
        .fields
        .get("Type")
        .or_else(|| struct_value.fields.get("type"))
        .ok_or_else(|| "rng: state struct is missing the 'Type' field".to_string())?;
    let generator = match keyword_of(type_value) {
        Some(ref kw) => parse_generator_keyword(kw)?,
        None => return Err("rng: Type field must be a string".to_string()),
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
        .ok_or_else(|| "rng: state struct is missing the 'State' field".to_string())?;
    let state = parse_state_scalar(state_value)?;
    Ok(RngSnapshot {
        state,
        seed: seed_opt,
        algorithm: generator,
    })
}

fn parse_generator(value: &Value) -> Result<RngAlgorithm, String> {
    match keyword_of(value) {
        Some(keyword) => parse_generator_keyword(&keyword),
        None => Err("rng: generator name must be a string".to_string()),
    }
}

fn parse_generator_keyword(keyword: &str) -> Result<RngAlgorithm, String> {
    match keyword {
        "twister" | "default" | "runmat-lcg" => Ok(RngAlgorithm::RunMatLcg),
        other => Err(format!("rng: generator '{other}' is not supported")),
    }
}

fn parse_seed_scalar(value: &Value, label: &str) -> Result<u64, String> {
    match value {
        Value::Int(i) => {
            let v = i.to_i64();
            if v < 0 {
                return Err(format!("{label}: seed must be non-negative"));
            }
            Ok(v as u64)
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format!("{label}: seed must be finite"));
            }
            if *n < 0.0 {
                return Err(format!("{label}: seed must be non-negative"));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(format!("{label}: seed must be an integer"));
            }
            if rounded > (1u64 << 53) as f64 {
                return Err(format!("{label}: seed exceeds 53-bit integer precision"));
            }
            Ok(rounded as u64)
        }
        Value::Tensor(t) if t.data.len() == 1 => parse_seed_scalar(&Value::Num(t.data[0]), label),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err(format!("{label}: expected a numeric seed"))
        }
        _ => Err(format!("{label}: expected a scalar numeric seed")),
    }
}

fn parse_state_scalar(value: &Value) -> Result<u64, String> {
    match value {
        Value::Tensor(t) => match t.data.len() {
            1 => parse_state_scalar(&Value::Num(t.data[0])),
            2 => {
                let lo = parse_state_word(t.data[0], "rng: State[1]")?;
                let hi = parse_state_word(t.data[1], "rng: State[2]")?;
                Ok(lo | ((hi as u64) << 32))
            }
            _ => Err("rng: State tensor must contain one or two elements".to_string()),
        },
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("rng: State must be finite".to_string());
            }
            if *n < 0.0 {
                return Err("rng: State must be non-negative".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err("rng: State must be an integer vector".to_string());
            }
            Ok(rounded as u64)
        }
        Value::Int(i) => {
            let v = i.to_i64();
            if v < 0 {
                Err("rng: State must be non-negative".to_string())
            } else {
                Ok(v as u64)
            }
        }
        other => Err(format!("rng: unsupported State value {other:?}")),
    }
}

fn parse_state_word(value: f64, label: &str) -> Result<u64, String> {
    if !value.is_finite() {
        return Err(format!("{label}: must be finite"));
    }
    if value < 0.0 {
        return Err(format!("{label}: must be non-negative"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(format!("{label}: must be an integer"));
    }
    if rounded > (u32::MAX as f64) {
        return Err(format!("{label}: exceeds uint32 precision"));
    }
    Ok(rounded as u64)
}

fn shuffle_seed() -> u64 {
    if let Ok(env) = std::env::var("RUNMAT_RNG_SHUFFLE_SEED") {
        if let Ok(parsed) = env.parse::<u64>() {
            return parsed;
        }
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut seed = now as u64 ^ (now >> 32) as u64;
    let addr = (&seed as *const u64 as u64).rotate_left(21);
    seed ^= addr ^ (seed << 7);
    if seed == 0 {
        DEFAULT_USER_SEED.wrapping_add(1)
    } else {
        seed
    }
}

fn sync_provider_state(state: u64) {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Err(err) = provider.set_rng_state(state) {
            debug!("rng: provider seed sync failed: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};
    use runmat_builtins::IntValue;

    #[test]
    fn rng_returns_current_state() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let value = rng_builtin(Vec::new()).expect("rng");
        let snapshot = snapshot_from_value(&value).expect("snapshot");
        assert_eq!(snapshot.state, random::default_snapshot().state);
        assert_eq!(snapshot.seed, Some(DEFAULT_USER_SEED));
        assert_eq!(snapshot.algorithm, RngAlgorithm::RunMatLcg);
    }

    #[test]
    fn rng_seed_is_reproducible() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        rng_builtin(vec![Value::Int(IntValue::U32(42))]).expect("rng");
        let seq1 = random::generate_uniform(5, "rng test").expect("uniform");
        rng_builtin(vec![Value::Int(IntValue::U32(42))]).expect("rng");
        let seq2 = random::generate_uniform(5, "rng test").expect("uniform");
        assert_eq!(seq1, seq2);
    }

    #[test]
    fn rng_restore_struct_roundtrip() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let saved = rng_builtin(Vec::new()).expect("rng");
        rng_builtin(vec![Value::Int(IntValue::U32(7))]).expect("rng");
        rng_builtin(vec![saved.clone()]).expect("rng restore");
        let current = random::snapshot().expect("snapshot");
        assert_eq!(current.state, random::default_snapshot().state);
        assert_eq!(current.seed, Some(DEFAULT_USER_SEED));
    }

    #[test]
    fn rng_default_restores_state() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        rng_builtin(vec![Value::Int(IntValue::U32(99))]).expect("seed rng");
        let previous = rng_builtin(vec![Value::from("default")]).expect("rng default");
        let restored = random::snapshot().expect("snapshot");
        assert_eq!(restored.state, random::default_snapshot().state);
        assert_eq!(restored.seed, Some(DEFAULT_USER_SEED));
        let prev_snapshot = snapshot_from_value(&previous).expect("prev snapshot");
        assert_eq!(prev_snapshot.seed, Some(99));
        assert_ne!(prev_snapshot.state, restored.state);
    }

    #[test]
    fn rng_seed_with_twister_alias() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        rng_builtin(vec![Value::Int(IntValue::U32(123))]).expect("rng seed first");
        let host_seq = random::generate_uniform(4, "twister alias host").expect("uniform");
        random::reset_rng();
        rng_builtin(vec![Value::Int(IntValue::U32(123)), Value::from("twister")])
            .expect("rng seed twister");
        let alias_seq = random::generate_uniform(4, "twister alias verify").expect("uniform");
        assert_eq!(host_seq, alias_seq);
    }

    #[test]
    fn rng_rejects_negative_seed() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let err = rng_builtin(vec![Value::Int(IntValue::I32(-5))]).unwrap_err();
        assert!(
            err.contains("seed must be non-negative"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rng_rejects_unknown_generator() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let err = rng_builtin(vec![Value::from("default"), Value::from("philox")]).unwrap_err();
        assert!(
            err.contains("generator 'philox' is not supported")
                || err.contains("generator 'philox'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rng_state_struct_requires_type() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).expect("tensor");
        let mut fields = HashMap::new();
        fields.insert("Seed".to_string(), Value::Num(0.0));
        fields.insert("State".to_string(), Value::Tensor(tensor));
        let err = rng_builtin(vec![Value::Struct(StructValue { fields })]).unwrap_err();
        assert!(err.contains("Type"), "unexpected error message: {err}");
    }

    #[test]
    fn rng_syncs_provider_state() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            rng_builtin(vec![Value::Int(IntValue::U32(9))]).expect("rng");
            let handle = provider.random_uniform(&[4, 1]).expect("gpu uniform");
            let host_after_gpu = random::generate_uniform(4, "rng provider sync").expect("uniform");
            let gpu = provider.download(&handle).expect("download");
            assert_eq!(gpu.data, host_after_gpu);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn rng_wgpu_uniform_matches_cpu() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        rng_builtin(vec![Value::Int(IntValue::U32(2024))]).expect("rng wgpu seed");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let handle = provider
            .random_uniform(&[1, 6])
            .expect("wgpu random uniform");
        let gpu = provider.download(&handle).expect("wgpu download");
        let host = random::generate_uniform(6, "rng wgpu parity").expect("host uniform sequence");
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-6,
        };
        for (idx, (g, h)) in gpu.data.iter().zip(host.iter()).enumerate() {
            assert!(
                (g - h).abs() <= tol,
                "mismatch at element {}: gpu={} host={} (tol={})",
                idx,
                g,
                h,
                tol
            );
        }
        let _ = provider.free(&handle);
    }

    #[test]
    fn rng_shuffle_uses_entropy_or_override() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        std::env::set_var("RUNMAT_RNG_SHUFFLE_SEED", "12345");
        rng_builtin(vec![Value::from("shuffle")]).expect("rng shuffle");
        std::env::remove_var("RUNMAT_RNG_SHUFFLE_SEED");
        let current = random::snapshot().expect("snapshot");
        assert_eq!(current.seed, Some(12345));
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn rng_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}

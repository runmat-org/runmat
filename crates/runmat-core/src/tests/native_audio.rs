//! Audio builtin execution through the real core interpreter and native tiers.

use std::path::Path;
use std::sync::Arc;

use futures::executor::block_on;
use runmat_filesystem::{replace_provider, MemoryFsProvider};
use runmat_jit::tiering::TieringConfig;
use runmat_value::{IntegerStorage, Value};

use crate::{
    ExecutableSource, InvocationControl, ProcedureInvocation, ProcedureTarget, RunError,
    RunMatSession,
};

const SOURCE: &str = r#"
function y = engineReadPcm(path)
  y = audioread(path, [2, 3], 'native');
end
function [y, rate] = engineReadFloat(path)
  [y, rate] = audioread(path, [1, 3], 'native');
end
function info = engineInfo(path)
  info = audioinfo(path);
end
function y = engineBadRange(path, samples)
  y = audioread(path, samples, 'native');
end
function engineDiscard(path)
  audioread(path);
  audioinfo(path);
end
function engineResetCount()
  global audioEngineCount
  audioEngineCount = 0;
end
function count = engineGetCount()
  global audioEngineCount
  count = audioEngineCount;
end
function [count, y] = engineReadNoReplay(path)
  global audioEngineCount
  audioEngineCount = audioEngineCount + 1;
  y = audioread(path, [2, 3], 'native');
  count = audioEngineCount;
end
function [count, info] = engineInfoNoReplay(path)
  global audioEngineCount
  audioEngineCount = audioEngineCount + 1;
  info = audioinfo(path);
  count = audioEngineCount;
end
"#;

fn wave(sample_rate: u32, channels: u16, bits: u16, format: u16, payload: &[u8]) -> Vec<u8> {
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let mut bytes = Vec::with_capacity(44 + payload.len());
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + u32::try_from(payload.len()).unwrap()).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&format.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn pcm16() -> Vec<u8> {
    let samples = [i16::MIN, i16::MAX, -4, 5, 10, -10];
    let payload = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    wave(22_050, 2, 16, 1, &payload)
}

fn float32() -> Vec<u8> {
    let samples = [
        f32::from_bits(0x8000_0000),
        0.25,
        f32::from_bits(0x7fc0_0042),
    ];
    let payload = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    wave(48_000, 1, 32, 3, &payload)
}

fn invocation(name: &str, path: &str, requested_outputs: usize) -> ProcedureInvocation {
    ProcedureInvocation {
        target: ProcedureTarget::Function(name.into()),
        arguments: crate::ProcedureArguments::Source(vec![Value::String(path.into())]),
        requested_outputs,
    }
}

fn check_audio_forms(
    session: &mut RunMatSession,
    unit: &crate::ExecutableUnit,
    control: &InvocationControl,
) {
    let pcm = block_on(session.invoke_executable(
        unit,
        invocation("engineReadPcm", "/engine/pcm.wav", 1),
        control,
    ))
    .expect("one-output PCM16 audio invocation");
    let [Value::Tensor(pcm)] = pcm.as_slice() else {
        panic!("expected one PCM16 tensor, got {pcm:?}");
    };
    assert_eq!(pcm.shape(), &[2, 2]);
    assert_eq!(
        pcm.integer_storage(),
        Some(&IntegerStorage::I16(vec![-4, 10, 5, -10]))
    );

    let float = block_on(session.invoke_executable(
        unit,
        invocation("engineReadFloat", "/engine/float.wav", 2),
        control,
    ))
    .expect("two-output Float32 audio invocation");
    let [Value::Tensor(float), Value::Num(rate)] = float.as_slice() else {
        panic!("expected Float32 tensor and sample rate, got {float:?}");
    };
    assert_eq!(float.shape(), &[3, 1]);
    assert_eq!(*rate, 48_000.0);
    assert_eq!(
        float
            .as_f32_slice()
            .unwrap()
            .iter()
            .map(|sample| sample.to_bits())
            .collect::<Vec<_>>(),
        vec![0x8000_0000, 0x3e80_0000, 0x7fc0_0042]
    );

    let metadata = block_on(session.invoke_executable(
        unit,
        invocation("engineInfo", "/engine/pcm.wav", 1),
        control,
    ))
    .expect("audioinfo invocation");
    let [Value::Struct(metadata)] = metadata.as_slice() else {
        panic!("expected audioinfo struct, got {metadata:?}");
    };
    assert_eq!(
        metadata.fields.get("Format"),
        Some(&Value::String("WAV".into()))
    );
    assert_eq!(metadata.fields.get("NumChannels"), Some(&Value::Num(2.0)));
    assert_eq!(
        metadata.fields.get("SampleRate"),
        Some(&Value::Num(22_050.0))
    );
    assert_eq!(metadata.fields.get("TotalSamples"), Some(&Value::Num(3.0)));
    assert_eq!(
        metadata.fields.get("BitsPerSample"),
        Some(&Value::Num(16.0))
    );

    let discarded = block_on(session.invoke_executable(
        unit,
        invocation("engineDiscard", "/engine/pcm.wav", 0),
        control,
    ))
    .expect("discarded audio invocations still execute");
    assert!(discarded.as_slice().is_empty());

    for (name, expected) in [
        ("engineReadPcm", "RunMat:audioread:Io"),
        ("engineInfo", "RunMat:audioinfo:Io"),
    ] {
        let error = block_on(session.invoke_executable(
            unit,
            invocation(name, "/engine/missing.wav", 1),
            control,
        ))
        .expect_err("missing file must preserve the owning builtin error");
        let RunError::Runtime(error) = error else {
            panic!("expected runtime error for {name}");
        };
        assert_eq!(error.identifier(), Some(expected));
    }
    let invalid_range = block_on(session.invoke_executable(
        unit,
        ProcedureInvocation {
            target: ProcedureTarget::Function("engineBadRange".into()),
            arguments: crate::ProcedureArguments::Source(vec![
                Value::String("/engine/pcm.wav".into()),
                Value::Tensor(runmat_value::Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap()),
            ]),
            requested_outputs: 1,
        },
        control,
    ))
    .expect_err("invalid range must keep the owning audio diagnostic");
    let RunError::Runtime(invalid_range) = invalid_range else {
        panic!("expected runtime audio range diagnostic");
    };
    assert_eq!(
        invalid_range.identifier(),
        Some("RunMat:audioread:InvalidArgument")
    );
}

fn check_native_audio_suspension_has_no_effect_replay(
    session: &mut RunMatSession,
    unit: &crate::ExecutableUnit,
) {
    let control = InvocationControl::default().force_generic_native();
    for (name, path) in [
        ("engineReadNoReplay", "/engine/pcm.wav"),
        ("engineInfoNoReplay", "/engine/pcm.wav"),
    ] {
        block_on(session.invoke_executable(
            unit,
            ProcedureInvocation {
                target: ProcedureTarget::Function("engineResetCount".into()),
                arguments: crate::ProcedureArguments::Source(Vec::new()),
                requested_outputs: 0,
            },
            &control,
        ))
        .expect("reset source effect count");
        let result = block_on(session.invoke_executable(unit, invocation(name, path, 2), &control))
            .expect("native audio call resumes once");
        assert_eq!(result.as_slice().first(), Some(&Value::Num(1.0)), "{name}");
        assert_eq!(result.as_slice().len(), 2, "{name}");
    }

    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(session.invoke_executable(
        unit,
        invocation("engineReadPcm", "/engine/pcm.wav", 1),
        &control.with_cancellation(cancellation.token()),
    ))
    .expect_err("cancelled native invocation must not publish audio");
    let RunError::Runtime(error) = error else {
        panic!("expected runtime cancellation");
    };
    assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
}

#[path = "native_audio/gated.rs"]
mod gated;

fn selected_jit_result(
    session: &mut RunMatSession,
    unit: &crate::ExecutableUnit,
    procedure: &str,
    path: &str,
    outputs: usize,
) -> runmat_runtime::execution::RootedValueSequence {
    // The counter changes only in invoke_tiered_procedure's Some(published)
    // branch. Isolating one preferred function per invocation proves that the
    // result being checked came from its selected native entry, not another
    // function's compilation or a merely hot tier site.
    for _ in 0..8 {
        let selected_before = session.stats().jit_compiled;
        let result = block_on(session.invoke_executable(
            unit,
            invocation(procedure, path, outputs),
            &InvocationControl::default(),
        ))
        .expect("tiered audio invocation");
        if session.stats().jit_compiled > selected_before {
            return result;
        }
    }
    panic!("{procedure} never executed a published native entry");
}

#[test]
fn audioinfo_and_audioread_execute_across_interpreter_generic_native_and_jit() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(MemoryFsProvider::new());
    provider
        .write_project_path(Path::new("/engine/pcm.wav"), &pcm16())
        .unwrap();
    provider
        .write_project_path(Path::new("/engine/float.wav"), &float32())
        .unwrap();
    let _owner = replace_provider(provider);
    runmat_gc::gc_test_context(|| {
        let mut interpreter = RunMatSession::with_options(false, false).unwrap();
        let interpreter_unit = block_on(interpreter.compile_executable_unit(
            ExecutableSource::new("audio-engine@1", "engineReadPcm.m", SOURCE),
            None,
        ))
        .unwrap();
        check_audio_forms(
            &mut interpreter,
            &interpreter_unit,
            &InvocationControl::default(),
        );
        let expected_pcm = block_on(interpreter.invoke_executable(
            &interpreter_unit,
            invocation("engineReadPcm", "/engine/pcm.wav", 1),
            &InvocationControl::default(),
        ))
        .unwrap();
        let expected_info = block_on(interpreter.invoke_executable(
            &interpreter_unit,
            invocation("engineInfo", "/engine/pcm.wav", 1),
            &InvocationControl::default(),
        ))
        .unwrap();

        let mut native = RunMatSession::with_options(false, false).unwrap();
        let native_unit = block_on(native.compile_executable_unit(
            ExecutableSource::new("audio-engine@1", "engineReadPcm.m", SOURCE),
            None,
        ))
        .unwrap();
        check_audio_forms(
            &mut native,
            &native_unit,
            &InvocationControl::default().force_generic_native(),
        );
        check_native_audio_suspension_has_no_effect_replay(&mut native, &native_unit);
        assert!(native.generic_native_cache_counts().0 > 0);

        let mut jit = RunMatSession::with_options(true, false).unwrap();
        jit.set_native_tiering_config_for_testing(TieringConfig {
            generic_hot_threshold: 1,
            specialized_hot_threshold: 2,
            deterministic: true,
            ..Default::default()
        });
        let jit_unit = block_on(jit.compile_executable_unit(
            ExecutableSource::new("audio-engine@1", "engineReadPcm.m", SOURCE),
            None,
        ))
        .unwrap();
        assert_eq!(
            selected_jit_result(&mut jit, &jit_unit, "engineReadPcm", "/engine/pcm.wav", 1),
            expected_pcm
        );
        let selected_float = selected_jit_result(
            &mut jit,
            &jit_unit,
            "engineReadFloat",
            "/engine/float.wav",
            2,
        );
        let [Value::Tensor(samples), Value::Num(rate)] = selected_float.as_slice() else {
            panic!("selected Float32 entry lost its two outputs");
        };
        assert_eq!(samples.shape(), &[3, 1]);
        assert_eq!(*rate, 48_000.0);
        assert_eq!(
            samples
                .as_f32_slice()
                .unwrap()
                .iter()
                .map(|sample| sample.to_bits())
                .collect::<Vec<_>>(),
            vec![0x8000_0000, 0x3e80_0000, 0x7fc0_0042]
        );
        assert_eq!(
            selected_jit_result(&mut jit, &jit_unit, "engineInfo", "/engine/pcm.wav", 1),
            expected_info
        );
    });
}

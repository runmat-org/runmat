//! Bounded explicit replay of an unaccepted volume-quality diagnostic.
//! Usage: volume_quality_replay INPUT.cbor CONTROLS.json SECONDS (1..=900).
//! Controls name the retained producer binding and original options; this tool
//! never creates a complete CAD-admitted solver mesh or qualification receipt.
use runmat_canonical_codec::{CanonicalCodecControl, CanonicalLimits};
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_tetrahedron::cdt::{
    replay_volume_quality, DelaunayVolumeMeshOptions, VolumeQualityBinding,
};
use std::{
    io::Read,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    binding: VolumeQualityBinding,
    options: DelaunayVolumeMeshOptions,
}
struct Deadline {
    started: Instant,
    duration: Duration,
    checkpoints: AtomicU64,
}
impl MeshingCancellationSignal for Deadline {
    fn is_cancelled(&self) -> bool {
        self.checkpoints.fetch_add(1, Ordering::Relaxed);
        self.started.elapsed() >= self.duration
    }
}
impl CanonicalCodecControl for Deadline {
    fn is_cancelled(&self) -> bool {
        MeshingCancellationSignal::is_cancelled(self)
    }
}
struct Events;
impl tracing::Subscriber for Events {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(std::collections::BTreeMap<String, String>);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name().into(), format!("{value:?}"));
            }
        }
        let mut fields = Fields(Default::default());
        event.record(&mut fields);
        eprintln!(
            "QUALITY_EVENT {}",
            serde_json::to_string(&fields.0).unwrap()
        );
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() != 3 {
        return Err("expected INPUT.cbor CONTROLS.json SECONDS".into());
    }
    let seconds: u64 = arguments[2].to_str().ok_or("invalid seconds")?.parse()?;
    if !(1..=900).contains(&seconds) {
        return Err("probe interval must be 1..=900 seconds".into());
    }
    let mut metadata = Vec::new();
    std::fs::File::open(&arguments[1])?
        .take(65537)
        .read_to_end(&mut metadata)?;
    if metadata.len() > 65536 {
        return Err("diagnostic metadata exceeds 64 KiB".into());
    }
    let metadata: Metadata = serde_json::from_slice(&metadata)?;
    let limits = CanonicalLimits::new(512 * 1024 * 1024, 50_000_000, 1024 * 1024, 64);
    let mut file = std::fs::File::open(&arguments[0])?;
    let length = usize::try_from(file.metadata()?.len())?;
    if length > limits.maximum_encoded_bytes {
        return Err("diagnostic exceeds byte ceiling".into());
    }
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)?;
    let mut extra = [0; 1];
    if file.read(&mut extra)? != 0 {
        return Err("diagnostic grew beyond byte ceiling".into());
    }
    let deadline = Deadline {
        started: Instant::now(),
        duration: Duration::from_secs(seconds),
        checkpoints: AtomicU64::new(0),
    };
    let result = tracing::subscriber::with_default(Events, || {
        replay_volume_quality(
            &bytes,
            metadata.binding,
            metadata.options,
            limits,
            &deadline,
            &deadline,
        )
    });
    match result {
        Ok(result) => println!(
            "QUALITY_PROBE {}",
            serde_json::json!({"status":"diagnostic-converged-not-whole-mesh-acceptance",
            "nodes":result.topology.nodes.len(),"cells":result.topology.tetrahedra.len(),
            "insertions":result.quality_mutations.iter().filter(|step| matches!(step, runmat_meshing_tetrahedron::cdt::DelaunayVolumeQualityMutation::Refined(_))).count(),"mutations":result.quality_mutations.iter().filter(|step| matches!(step, runmat_meshing_tetrahedron::cdt::DelaunayVolumeQualityMutation::Optimized(_))).count(),
            "elapsed_seconds":deadline.started.elapsed().as_secs_f64(),"checkpoints":deadline.checkpoints.load(Ordering::Relaxed)})
        ),
        Err(error) => println!(
            "QUALITY_PROBE {}",
            serde_json::json!({"status":"diagnostic-stopped-not-acceptance","error":error.to_string(),
            "elapsed_seconds":deadline.started.elapsed().as_secs_f64(),"checkpoints":deadline.checkpoints.load(Ordering::Relaxed)})
        ),
    }
    Ok(())
}

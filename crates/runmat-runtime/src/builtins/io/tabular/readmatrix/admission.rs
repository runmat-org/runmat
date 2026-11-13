//! One typed control identity for staged and final readmatrix selection.
//! Resource evidence is supplied by the installed host owner, not by this
//! control digest or a filename size estimate.

use super::{Delimiter, OutputTemplate, RangeSpec, ReadMatrixOptions, BUILTIN_NAME};
use crate::context::host_effect::{
    admit_host_effect_stage, HostEffectAllocationError, HostEffectOutput, HostEffectRequest,
    HostEffectStagePermit, HostEffectWorkload,
};
use crate::context::RuntimeContext;
use runmat_execution::Digest;
use runmat_value::{NumericDType, NumericScalar};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use runmat_filesystem::FsProvider;

pub(super) fn digest_serializable<T: serde::Serialize>(facts: &T) -> Digest {
    crate::context::host_effect::hash_serializable(facts)
}

#[derive(serde::Serialize)]
enum OutputFact {
    Numeric(NumericDType),
    Logical,
}

#[derive(serde::Serialize)]
struct ScalarFact {
    dtype: NumericDType,
    exact_bits: [u8; 8],
}

impl From<NumericScalar> for ScalarFact {
    fn from(value: NumericScalar) -> Self {
        let exact_bits = match value {
            NumericScalar::F64(value) => value.to_bits().to_le_bytes(),
            NumericScalar::F32(value) => u64::from(value.to_bits()).to_le_bytes(),
            NumericScalar::I8(value) => (value as i64 as u64).to_le_bytes(),
            NumericScalar::I16(value) => (value as i64 as u64).to_le_bytes(),
            NumericScalar::I32(value) => (value as i64 as u64).to_le_bytes(),
            NumericScalar::I64(value) => (value as u64).to_le_bytes(),
            NumericScalar::U8(value) => u64::from(value).to_le_bytes(),
            NumericScalar::U16(value) => u64::from(value).to_le_bytes(),
            NumericScalar::U32(value) => u64::from(value).to_le_bytes(),
            NumericScalar::U64(value) => value.to_le_bytes(),
        };
        Self {
            dtype: value.numeric_dtype(),
            exact_bits,
        }
    }
}

#[derive(serde::Serialize)]
struct ControlFacts<'a> {
    delimiter: Option<&'a Delimiter>,
    num_header_lines: usize,
    decimal_separator: char,
    thousands_separator: Option<char>,
    treat_as_missing: &'a BTreeSet<String>,
    empty_value: Option<ScalarFact>,
    range: Option<&'a RangeSpec>,
    output: OutputFact,
    output_type_seen: bool,
    encoding: &'a str,
}

/// A live `Like` prototype needs resident class/storage/readback facts before
/// it can have a complete finite identity. Ordinary unbudgeted calls keep
/// their existing materialization route.
pub(super) fn controls_digest(options: &ReadMatrixOptions) -> Option<Digest> {
    let output = match &options.output_template {
        OutputTemplate::Numeric(dtype) => OutputFact::Numeric(*dtype),
        OutputTemplate::Logical => OutputFact::Logical,
        OutputTemplate::Like(_) => return None,
    };
    let facts = ControlFacts {
        delimiter: options.delimiter.as_ref(),
        num_header_lines: options.num_header_lines,
        decimal_separator: options.decimal_separator,
        thousands_separator: options.thousands_separator,
        treat_as_missing: &options.treat_as_missing,
        empty_value: options.empty_value.map(ScalarFact::from),
        range: options.range.as_ref(),
        output,
        output_type_seen: options.output_type_seen,
        encoding: &options.encoding,
    };
    Some(digest_serializable(&facts))
}

/// The common host String/no-controls form can be identified before any
/// option or path allocation. This must match the parsed default controls.
pub(super) fn default_controls_digest() -> Digest {
    let missing = BTreeSet::new();
    digest_serializable(&ControlFacts {
        delimiter: None,
        num_header_lines: 0,
        decimal_separator: '.',
        thousands_separator: None,
        treat_as_missing: &missing,
        empty_value: None,
        range: None,
        output: OutputFact::Numeric(NumericDType::F64),
        output_type_seen: false,
        encoding: "utf-8",
    })
}

pub(super) fn request<'a>(
    path: &'a Path,
    filesystem: &'a Arc<dyn FsProvider>,
    controls: Digest,
    domain_facts: Digest,
) -> HostEffectRequest<'a> {
    HostEffectRequest {
        operation: BUILTIN_NAME,
        workload: HostEffectWorkload::ReadMatrixText {
            source: path,
            controls,
            output: HostEffectOutput::ReturnedValue,
        },
        domain_facts,
        filesystem,
    }
}

pub(super) fn finite_budget(runtime: Option<&RuntimeContext>) -> bool {
    runtime.is_some_and(|runtime| {
        runtime.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        })
    })
}

/// The caller's absolute host String is borrowed through initial selection.
/// No PathBuf, parsed option object, or digest serialization buffer is made
/// before the installed authority reserves its pre-open envelope.
pub(super) fn admit_default_absolute(
    runtime: &RuntimeContext,
    path: &Path,
    filesystem: &Arc<dyn FsProvider>,
) -> crate::BuiltinResult<HostEffectStagePermit> {
    let controls = default_controls_digest();
    admit_host_effect_stage(runtime, request(path, filesystem, controls, controls))
}

pub(super) fn allocation_error(
    operation: &'static str,
    source: HostEffectAllocationError,
) -> crate::RuntimeError {
    let identifier = match source {
        HostEffectAllocationError::Cancelled => "RunMat:Cancelled",
        HostEffectAllocationError::Revoked | HostEffectAllocationError::InvalidInvocation => {
            "RunMat:Placement:StaleResourceAuthority"
        }
        HostEffectAllocationError::Unavailable
        | HostEffectAllocationError::Overflow
        | HostEffectAllocationError::ExceedsStageEnvelope
        | HostEffectAllocationError::ExceedsCapacity
        | HostEffectAllocationError::AllocationFailed => "RunMat:Placement:ResourceLimit",
    };
    crate::build_runtime_error(format!("{operation}: admitted allocation failed: {source}"))
        .with_builtin(operation)
        .with_identifier(identifier)
        .with_source(source)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;

    struct Finite(ManagedHostEffectLedger);

    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn control_identity_is_order_independent_but_exact_for_numeric_bits() {
        let mut first = ReadMatrixOptions::default();
        first.add_missing_token("NA");
        first.add_missing_token("N/A");
        let mut reordered = ReadMatrixOptions::default();
        reordered.add_missing_token("N/A");
        reordered.add_missing_token("NA");
        assert_eq!(controls_digest(&first), controls_digest(&reordered));
        reordered.empty_value = Some(NumericScalar::F64(-0.0));
        first.empty_value = Some(NumericScalar::F64(0.0));
        assert_ne!(controls_digest(&first), controls_digest(&reordered));
        assert_eq!(
            controls_digest(&ReadMatrixOptions::default()),
            Some(default_controls_digest())
        );
        reordered.empty_value = Some(NumericScalar::U64(u64::MAX));
        assert_ne!(controls_digest(&first), controls_digest(&reordered));
    }

    #[test]
    fn finite_public_default_fails_before_missing_source_without_installed_evidence() {
        #[cfg(windows)]
        let absent = "C:\\readmatrix-not-opened.csv";
        #[cfg(not(windows))]
        let absent = "/readmatrix-not-opened.csv";
        let ledger = ManagedHostEffectLedger::new_with_epoch(1024, 7);
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
                );
        let error = futures::executor::block_on(super::super::readmatrix_builtin(
            Some(runtime),
            runmat_value::Value::from(absent),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("missing stage evidence must deny before source open");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

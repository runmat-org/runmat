//! Process-local memo of rederived fluid recovery and state-publication resource plans.
//!
//! Each prepared plan is a pure function of its request, its complete input objects and the
//! output policy, and every stage of one study (preparation, worker authority checks, import)
//! asks for the same plan again. Rederiving it decodes and re-prepares the original fluid source
//! each time. The memo keys each derivation by a digest of the request, the policy and every
//! input object's descriptor and actual bytes (each chunk is rehashed on every call, so a hit
//! still authenticates the exact content), and keeps one plan per phase kind, like the fluid
//! resource projection cache. Failed derivations are never memoized.
use crate::*;
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{Digest, DigestStream};
use std::sync::Mutex;

pub(crate) struct PlanMemo(Mutex<Option<(Digest, FeaResourcePlan)>>);

impl PlanMemo {
    pub(crate) const fn new() -> Self {
        Self(Mutex::new(None))
    }

    pub(crate) fn get_or_derive(
        &self,
        domain: &str,
        request: &[u8],
        inputs: &[PreparedAnalysisObjects],
        policy: AnalysisOutputPolicy,
        derive: impl FnOnce() -> AnalysisBridgeResult<FeaResourcePlan>,
    ) -> AnalysisBridgeResult<FeaResourcePlan> {
        let key = memo_key(domain, request, inputs, policy);
        if let Some((cached, plan)) = self.0.lock().ok().as_deref().and_then(Option::as_ref) {
            if *cached == key {
                return Ok(plan.clone());
            }
        }
        let plan = derive()?;
        if let Ok(mut last) = self.0.lock() {
            *last = Some((key, plan.clone()));
        }
        Ok(plan)
    }
}

fn memo_key(
    domain: &str,
    request: &[u8],
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> Digest {
    let mut stream = DigestStream::new();
    let mut field = |bytes: &[u8]| {
        stream.update(&(bytes.len() as u64).to_le_bytes());
        stream.update(bytes);
    };
    field(domain.as_bytes());
    field(request);
    field(format!("{policy:?}").as_bytes());
    field(&(inputs.len() as u64).to_le_bytes());
    for input in inputs {
        field(format!("{:?}", input.manifest).as_bytes());
        field(format!("{:?}", input.root).as_bytes());
        field(&(input.objects.len() as u64).to_le_bytes());
        for object in &input.objects {
            field(format!("{:?}", object.descriptor).as_bytes());
            field(Digest::sha256(&object.bytes).bytes());
        }
    }
    stream.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memo_key_binds_request_policy_and_object_bytes() {
        let limits = runmat_execution_artifact::object::ObjectInventoryLimits::default();
        let policy = AnalysisOutputPolicy::new(1 << 20, limits).unwrap();
        let bytes = b"fluid-source-bytes".to_vec();
        let objects = |payload: &[u8]| {
            vec![prepare_analysis_objects(
                runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh,
                Digest::sha256(payload),
                payload,
                1 << 20,
                runmat_execution_artifact::ObjectNamespace::InputValue,
                limits,
            )
            .unwrap()]
        };
        let base = memo_key("fluid", b"request", &objects(&bytes), policy);
        assert_eq!(base, memo_key("fluid", b"request", &objects(&bytes), policy));
        assert_ne!(base, memo_key("fluid", b"other-request", &objects(&bytes), policy));
        assert_ne!(base, memo_key("state", b"request", &objects(&bytes), policy));
        let mut tampered = objects(&bytes);
        let last = tampered[0].objects.len() - 1;
        tampered[0].objects[last].bytes[0] ^= 1;
        assert_ne!(base, memo_key("fluid", b"request", &tampered, policy));
    }
}

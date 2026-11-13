//! Independent canonical-request tests shared across actual structural fixtures.
use super::*;
fn peak(demand: &mut FeaResourceDemand) -> &mut ByteCount {
    match demand {
        FeaResourceDemand::StructuralModalExtraction(d) => &mut d.constraint_compilation_peak_bytes,
        FeaResourceDemand::StructuralFrequencyEntry(d)
        | FeaResourceDemand::StructuralFrequencyFieldRecovery(d) => {
            &mut d.constraint_compilation_peak_bytes
        }
        _ => panic!("compiler-bearing fixture required"),
    }
}
pub(super) fn assert_compiler_contract(plan: &FeaResourcePlan, original_peak: u64) {
    let mut input = plan.input.clone();
    assert_eq!(peak(&mut input.demand).0, original_peak);
    assert!(original_peak > 1);
    peak(&mut input.demand).0 -= 1;
    assert_ne!(build_resource_plan(input).unwrap().identity, plan.identity);
    let mut input = plan.input.clone();
    peak(&mut input.demand).0 = 0;
    assert!(build_resource_plan(input).is_err());
    let mut encoded = serde_json::to_value(plan).unwrap();
    fn remove(value: &mut serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(fields) => {
                fields.remove("constraint_compilation_peak_bytes").is_some()
                    || fields.values_mut().any(remove)
            }
            serde_json::Value::Array(items) => items.iter_mut().any(remove),
            _ => false,
        }
    }
    assert!(remove(&mut encoded));
    assert!(serde_json::from_value::<FeaResourcePlan>(encoded).is_err());
    let mut input = plan.input.clone();
    let compiler = plan.request.memory_bytes * 2;
    peak(&mut input.demand).0 = compiler;
    input.accuracy_ceiling.maximum_resident_bytes = None;
    let dominated = build_resource_plan(input.clone()).unwrap();
    assert_eq!(
        dominated.request.memory_bytes,
        compiler
            + input.worker.fixed_memory_bytes.0
            + dominated.breakdown.input_transport_closure_bytes.0
            + dominated.breakdown.input_decoded_resident_bytes.0
            + dominated.breakdown.resource_plan_transport_closure_bytes.0
            + dominated.breakdown.resource_plan_decoded_resident_bytes.0
    );
    input.accuracy_ceiling.maximum_resident_bytes = Some(dominated.request.memory_bytes);
    assert_eq!(
        build_resource_plan(input.clone()).unwrap().request,
        dominated.request
    );
    input.accuracy_ceiling.maximum_resident_bytes =
        Some(input.accuracy_ceiling.resident_byte_budget() - 1);
    assert!(matches!(
        build_resource_plan(input),
        Err(ResourcePlanningError::AccuracyCeiling { .. })
    ));
}

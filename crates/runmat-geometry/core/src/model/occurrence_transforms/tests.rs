use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct Control {
    bytes: AtomicUsize,
    iterations: AtomicUsize,
    limit: Option<usize>,
}
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        self.iterations.fetch_add(count as usize, Ordering::Relaxed);
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        if self.limit.is_some_and(|limit| bytes as usize > limit) {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                "fixture budget",
            ));
        }
        self.bytes.fetch_add(bytes as usize, Ordering::Relaxed);
        Ok(())
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Prepared occurrence matrices match original ancestor composition bit for bit for
//   repeated and nested placements. Repeated queries do not repeat preparation allocation or work
//   charges. This covers occurrence lookup, not all geometric queries or whole-model memory.
#[test]
fn prepared_world_lookup_matches_original_composition_for_repeated_and_nested_instances() {
    let mut topology = crate::model::exact_topology_tests::topology();
    let template_assembly = topology
        .assemblies
        .iter()
        .find(|a| a.id != topology.root_assembly_id)
        .unwrap()
        .clone();
    let template_instance = topology.instances[0].clone();
    let mut parent = topology.root_assembly_id.clone();
    for ordinal in 0..512 {
        let mut assembly = template_assembly.clone();
        assembly.id.source_topology_id = format!("prepared-{ordinal}");
        assembly.id.assembly_path = parent
            .assembly_path
            .iter()
            .cloned()
            .chain(std::iter::once(format!("child-{ordinal}")))
            .collect();
        assembly.body_ids.clear();
        assembly.child_instance_ids.clear();
        let mut instance = template_instance.clone();
        instance.id.source_topology_id = format!("placement-{ordinal}");
        instance.id.assembly_path = assembly.id.assembly_path.clone();
        instance.parent_assembly_id = parent.clone();
        instance.instantiated_assembly_id = assembly.id.clone();
        instance.transform.0[3] = ordinal as f64 * 0.1;
        instance.transform.0[7] = -(ordinal as f64) * 0.003;
        parent = if ordinal < 32 {
            assembly.id.clone()
        } else {
            topology.root_assembly_id.clone()
        };
        topology.assemblies.push(assembly);
        topology.instances.push(instance);
    }
    topology.assemblies.reverse();
    topology.instances.reverse();
    let control = Control::default();
    let prepared = PreparedOccurrenceTransforms::new(&topology, &control).unwrap();
    assert_eq!(
        control.bytes.load(Ordering::Relaxed),
        PreparedOccurrenceTransforms::requested_allocation_bytes(&topology).unwrap()
    );
    for assembly in &topology.assemblies {
        assert_eq!(
            prepared
                .world_transform_for(&assembly.id)
                .unwrap()
                .0
                .map(f64::to_bits),
            topology
                .world_transform_for(&assembly.id)
                .unwrap()
                .0
                .map(f64::to_bits)
        );
    }
    let before = (
        control.bytes.load(Ordering::Relaxed),
        control.iterations.load(Ordering::Relaxed),
    );
    for _ in 0..10000 {
        prepared
            .world_transform_for(&topology.assemblies[0].id)
            .unwrap();
    }
    assert_eq!(
        before,
        (
            control.bytes.load(Ordering::Relaxed),
            control.iterations.load(Ordering::Relaxed)
        )
    );
    let mut unknown = topology.root_assembly_id.clone();
    unknown.assembly_path.push("absent".into());
    assert!(prepared.world_transform_for(&unknown).is_err());
}

#[test]
fn prepared_occurrences_reject_cycles_and_admit_storage_before_work() {
    let mut topology = crate::model::exact_topology_tests::topology();
    let denied = Control {
        limit: Some(0),
        ..Control::default()
    };
    assert!(matches!(
        PreparedOccurrenceTransforms::new(&topology, &denied),
        Err(GeometryEvaluationError {
            kind: GeometryEvaluationErrorKind::AllocationBudgetExceeded,
            ..
        })
    ));
    assert_eq!(denied.iterations.load(Ordering::Relaxed), 0);
    topology.instances[0].parent_assembly_id =
        topology.instances[0].instantiated_assembly_id.clone();
    assert!(matches!(
        PreparedOccurrenceTransforms::new(&topology, &Control::default()),
        Err(GeometryEvaluationError {
            kind: GeometryEvaluationErrorKind::InconsistentGeometry,
            ..
        })
    ));
}

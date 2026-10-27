//! Named field intent is frozen before mesh resolution; no live source or facts on reopen.
use super::*;
use runmat_analysis_execution::FrozenDeferredSignal;

impl SignalFixture {
    pub(super) async fn prepare_named_field(
        authored: PhysicalSignal,
        explicit: PhysicalSignal,
        scale: Vec<f64>,
        mesh: &runmat_meshing_core::SolverMeshArtifact,
    ) -> Self {
        let width = authored.shape().topology_component_count().unwrap();
        let nodes = mesh.topology.nodes.len();
        let source = format!("function y = pulse(t)\narguments\n t (:,1) double\nend\nscale = [{}];\ny = reshape(t * scale, numel(t), {nodes}, {width});\nend\n",
            scale.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "));
        let compilation::CompiledSignalFixture {
            session: _session,
            project_root,
            project,
            revision,
            unit,
            owner_identity,
        } = compilation::compile(&source).await;
        let executable = unit.portable_envelope_for(Some("pulse")).unwrap();
        let frozen = FrozenDeferredSignal::prepare(
            authored.clone(),
            revision.clone(),
            BundleCallable {
                owner_identity,
                qualified_name: "loads.pulse".into(),
                source_digest: Digest::sha256(&source),
            },
            executable,
            BTreeSet::new(),
            4096 + 2 * nodes * width * 16,
        )
        .unwrap();
        assert_eq!(frozen.authored(), &authored);
        assert_eq!(
            frozen
                .authored()
                .shape()
                .realized_component_count(Some(nodes)),
            None
        );
        let frozen_identity = frozen.identity().unwrap();
        let reopened: FrozenDeferredSignal =
            serde_json::from_slice(&serde_json::to_vec(&frozen).unwrap()).unwrap();
        assert_eq!(reopened.identity().unwrap(), frozen_identity);
        reopened.validate_for(&frozen).unwrap();
        let resolved = reopened.resolve(mesh).unwrap();
        reopened
            .validate_resolved_expectation(&resolved.prepared().expectation().unwrap(), mesh)
            .unwrap();
        assert_ne!(
            resolved.prepared().admitted.materialization_identity,
            frozen.admitted().materialization_identity
        );
        assert_eq!(
            resolved.prepared().admitted.generator_identity,
            frozen.admitted().generator_identity
        );
        let mut foreign = mesh.clone();
        {
            let quantity_change = 1.0;
            let quantity_index = 0;
            let quantity_slot = &mut foreign.topology.nodes[0].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] += quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        assert!(reopened.resolve(&foreign).is_err());
        let other_order = if mesh.topology.nodes.len() == 4 {
            runmat_meshing_core::ElementOrder::Tet10
        } else {
            runmat_meshing_core::ElementOrder::Tet4
        };
        let foreign = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(other_order);
        assert!(reopened
            .validate_resolved_expectation(&resolved.prepared().expectation().unwrap(), &foreign)
            .is_err());
        Self {
            _project_root: project_root,
            project,
            revision,
            prepared: resolved.into_prepared(),
            explicit,
        }
    }
}

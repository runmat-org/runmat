//! Solved bodies in a result scene: region tags, hidden and context bodies, extremes over the
//! bodies drawn in field colour, and a study's fixed deformation scale.
use super::regions::{BodyRegions, ViewRegions, BODY_REGION_TAG};
use super::*;
use crate::analysis::{AnalysisRenderMesh, AnalysisRenderTopologySource};
use std::sync::Arc;

/// Four boundary triangles, one per volume element: elements 0 and 3 are the arm, 1 the
/// payload stand-in and 2 the base.
fn yaw_arm() -> (AnalysisRenderTopology, Arc<BodyRegions>) {
    let topology = AnalysisRenderTopology {
        schema_version: "test".into(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        volume_element_count: 4,
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "mesh".into(),
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [1, 3, 2], [0, 2, 3], [0, 1, 3]],
            regions: vec![],
            vertex_volume_node_indices: vec![Some(0), Some(1), Some(2), Some(3)],
            triangle_volume_element_indices: vec![Some(0), Some(1), Some(2), Some(3)],
        }],
    };
    let bodies = BodyRegions::of_elements(
        &["arm", "payload", "base"],
        vec![Some(0), Some(1), Some(2), Some(0)],
    );
    (topology, Arc::new(bodies))
}

fn element_stress() -> FieldData {
    FieldData::inline(
        AnalysisFieldLocation::Element,
        1,
        vec![10.0, 50.0, 30.0, 20.0],
        "structural.von_mises",
        "Pa",
    )
}

fn draw(
    topology: &AnalysisRenderTopology,
    field: &FieldData,
    regions: ViewRegions,
) -> GeometryScene {
    scene_from_topology_with_stats(
        "run",
        topology,
        Some(field),
        None,
        "undeformed",
        1.0,
        false,
        Some(SceneStats {
            element_count: topology.volume_element_count,
            regions,
            ..SceneStats::default()
        }),
    )
    .unwrap()
    .0
}

fn surfaces(scene: &GeometryScene) -> Vec<&GeometrySceneChunk> {
    scene
        .chunks
        .iter()
        .filter(|chunk| chunk.render_data.pipeline_type == PipelineType::Triangles)
        .collect()
}

fn range(scene: &GeometryScene) -> (f64, f64) {
    let legend = scene
        .overlay
        .as_ref()
        .unwrap()
        .scalar_legend
        .as_ref()
        .unwrap();
    (legend.minimum, legend.maximum)
}

#[test]
fn each_solved_body_carries_its_study_region_through_a_saved_scene() {
    let (topology, bodies) = yaw_arm();
    let scene = draw(
        &topology,
        &element_stress(),
        ViewRegions::of_display(Some(bodies), None),
    );
    let tagged: Vec<_> = surfaces(&scene)
        .iter()
        .map(|chunk| {
            let [region] = chunk.regions.as_slice() else {
                panic!("one region per body chunk: {:?}", chunk.regions);
            };
            assert_eq!(region.tag.as_deref(), Some(BODY_REGION_TAG));
            assert_eq!(
                chunk.owner_node_ids,
                std::slice::from_ref(&region.region_id)
            );
            (region.region_id.clone(), chunk.triangle_count())
        })
        .collect();
    assert_eq!(
        tagged,
        [
            ("arm".to_string(), 2),
            ("payload".to_string(), 1),
            ("base".to_string(), 1)
        ]
    );
    let overlay = scene.overlay.as_ref().unwrap();
    assert_eq!(overlay.region_count, 3);
    // Every body is drawn: the extremes span all of them.
    assert_eq!(range(&scene), (10.0, 50.0));
    // The tags and owners survive the saved scene a viewer replays.
    let replayed = runmat_plot::event::FigureScene::from_geometry_scene(&scene)
        .into_geometry_scene("replay", 1)
        .unwrap();
    let payload = replayed
        .chunks
        .iter()
        .find(|chunk| {
            chunk
                .regions
                .iter()
                .any(|region| region.region_id == "payload")
        })
        .expect("the payload's surface keeps its region");
    assert_eq!(payload.owner_node_ids, ["payload"]);
    assert!(replayed.chunks.iter().any(|chunk| {
        chunk.render_data.pipeline_type == PipelineType::Lines
            && chunk.owner_node_ids == ["arm"]
            && chunk
                .segment_regions
                .iter()
                .any(|region| region.region_id == "arm")
    }));
}

#[test]
fn hidden_and_context_bodies_leave_the_extremes_to_the_bodies_in_field_colour() {
    let (topology, bodies) = yaw_arm();
    let display = runmat_analysis_core::AuthoredOutputDisplay {
        hide: vec!["payload".into()],
        context: vec!["base".into()],
        ..runmat_analysis_core::AuthoredOutputDisplay::unit_only(
            "von_mises",
            runmat_analysis_core::DisplayUnitSource::Si,
        )
    };
    let regions = ViewRegions::of_display(Some(bodies), Some(&display));
    let scene = draw(&topology, &element_stress(), regions.clone());
    // The payload's 50 and the base's 30 are left out: the arm alone sets the range.
    assert_eq!(range(&scene), (10.0, 20.0));
    let drawn: Vec<_> = surfaces(&scene)
        .iter()
        .map(|chunk| chunk.regions[0].region_id.clone())
        .collect();
    assert_eq!(drawn, ["arm", "base"]);
    let base = surfaces(&scene)[1];
    assert!(base
        .vertices
        .iter()
        .all(|vertex| vertex.color == bodies::CONTEXT_COLOR));
    assert_eq!(
        base.render_data.material.alpha_mode,
        runmat_plot::core::AlphaMode::Blend
    );
    let overlay = scene.overlay.as_ref().unwrap();
    let kinds: Vec<_> = overlay
        .regions
        .iter()
        .map(|region| (region.region_id.as_str(), region.kind.as_deref()))
        .collect();
    assert_eq!(kinds, [("arm", Some("body")), ("base", Some("context"))]);
    // The legend's true extremes are counted over the same bodies: elements 0 and 3, and the
    // nodes of those elements.
    let counted = regions
        .counted_entities(&[vec![0, 1, 2], vec![1, 3, 2], vec![0, 2, 3], vec![0, 1, 3]])
        .unwrap();
    assert_eq!(counted.elements, [true, false, false, true]);
    assert_eq!(counted.nodes, [true, true, true, true]);
    // A display naming a region no body carries warns instead of failing the view.
    let mut unknown = regions;
    unknown.hide.push("bracket".into());
    let scene = draw(&topology, &element_stress(), unknown);
    assert!(scene.overlay.as_ref().unwrap().warnings[0].contains("'bracket'"));
}

#[test]
fn a_study_deformation_scale_exaggerates_every_run_equally() {
    let (topology, _) = yaw_arm();
    let displacement = |peak: f64| {
        FieldData::inline(
            AnalysisFieldLocation::Node,
            3,
            vec![
                0.0, 0.0, 0.0, peak, 0.0, 0.0, 0.0, peak, 0.0, 0.0, 0.0, peak,
            ],
            "structural.displacement",
            "m",
        )
    };
    let scale = |peak: f64, fixed: Option<f64>| {
        scene_from_topology_with_stats(
            "run",
            &topology,
            None,
            Some(&displacement(peak)),
            "exaggerated",
            1.0,
            false,
            Some(SceneStats {
                element_count: topology.volume_element_count,
                fixed_deformation: fixed,
                ..SceneStats::default()
            }),
        )
        .unwrap()
        .1
    };
    // Each run's automatic scale follows its own displacement.
    assert_ne!(scale(1e-3, None), scale(2e-3, None));
    // The study's scale is the same for both runs.
    assert_eq!(scale(1e-3, Some(200.0)), 200.0);
    assert_eq!(scale(2e-3, Some(200.0)), 200.0);
}

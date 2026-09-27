use super::ProviderWorkload;

#[test]
fn scalar_work_is_optional_in_older_queries_and_round_trips_exactly() {
    let mut workload: ProviderWorkload =
        serde_json::from_str(r#"{"elements":3,"flops":null,"batch":null}"#).unwrap();
    assert_eq!(workload.scalar_work_units, None);
    assert_eq!(workload.preparation_route, None);
    for units in [0, 1, u64::MAX] {
        workload.scalar_work_units = Some(units);
        let bytes = serde_json::to_vec(&workload).unwrap();
        assert_eq!(
            serde_json::from_slice::<ProviderWorkload>(&bytes).unwrap(),
            workload
        );
    }
}

#[test]
fn preparation_route_is_explicit_and_round_trips_without_shader_payload() {
    use crate::{ProviderPrecision, ProviderPreparationProgram, ProviderPreparationRoute};
    assert_eq!(
        ProviderPreparationProgram::Elementwise {
            shader: "private source"
        }
        .route(),
        None
    );
    for target in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let workload = ProviderWorkload {
            preparation_route: ProviderPreparationProgram::FloatingCast { target }.route(),
            ..Default::default()
        };
        assert_eq!(
            workload.preparation_route,
            Some(ProviderPreparationRoute::FloatingCast { target })
        );
        let encoded = serde_json::to_vec(&workload).unwrap();
        assert_eq!(
            serde_json::from_slice::<ProviderWorkload>(&encoded).unwrap(),
            workload
        );
    }
}

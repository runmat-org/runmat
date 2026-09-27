use super::ProviderWorkload;

#[test]
fn scalar_work_is_optional_in_older_queries_and_round_trips_exactly() {
    let mut workload: ProviderWorkload =
        serde_json::from_str(r#"{"elements":3,"flops":null,"batch":null}"#).unwrap();
    assert_eq!(workload.scalar_work_units, None);
    for units in [0, 1, u64::MAX] {
        workload.scalar_work_units = Some(units);
        let bytes = serde_json::to_vec(&workload).unwrap();
        assert_eq!(
            serde_json::from_slice::<ProviderWorkload>(&bytes).unwrap(),
            workload
        );
    }
}

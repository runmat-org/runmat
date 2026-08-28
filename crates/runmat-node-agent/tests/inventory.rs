#[test]
fn inventory_is_normalized_and_names_runtime_and_platform_capabilities() {
    let inventory = runmat_node_agent::inventory::collect().unwrap();
    assert!(inventory.cpu_millicores >= 1_000);
    assert!(inventory.memory_bytes > 0);
    assert!(inventory.capabilities.contains_key("runmat.version"));
    assert!(inventory.capabilities.contains_key("os"));
    assert!(inventory.capabilities.contains_key("arch"));
    assert!(inventory
        .capabilities
        .contains_key("platform.process-isolation"));
}

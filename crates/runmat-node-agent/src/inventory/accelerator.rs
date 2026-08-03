use std::collections::BTreeMap;

pub(super) struct AcceleratorInventory {
    pub count: u32,
    pub class: Option<String>,
    pub memory_bytes: u64,
}

pub(super) fn inventory() -> AcceleratorInventory {
    let class = std::env::var("RUNMAT_NODE_ACCELERATOR_CLASS")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let count = std::env::var("RUNMAT_NODE_ACCELERATOR_COUNT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| u32::from(class.is_some()));
    let memory_bytes = std::env::var("RUNMAT_NODE_ACCELERATOR_MEMORY_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    AcceleratorInventory {
        count,
        class,
        memory_bytes,
    }
}

pub(super) fn capabilities() -> BTreeMap<String, String> {
    let mut values = BTreeMap::new();
    if let Some(class) = inventory().class {
        values.insert("accelerator.class".to_string(), class);
    }
    values
}

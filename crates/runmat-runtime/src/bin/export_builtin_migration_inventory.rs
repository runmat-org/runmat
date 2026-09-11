fn main() -> Result<(), Box<dyn std::error::Error>> {
    print!(
        "{}",
        runmat_runtime::builtin::migration_inventory::migration_inventory_json()?
    );
    Ok(())
}

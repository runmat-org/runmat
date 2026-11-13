//! Check portable geometry closure one authenticated tile at a time. Shared
//! storage and tile codecs own identity and shape validation, not the CLI.
use anyhow::{anyhow, ensure, Context, Result};
use runmat_runtime::analysis::storage;

pub(super) fn verify_saved_tiles(run_id: &str) -> Result<serde_json::Value> {
    let index = storage::load_render_tile_index(run_id)
        .map_err(|error| anyhow!(error))?
        .context("snapshot has no saved render topology")?;
    index.validate().map_err(|error| anyhow!(error))?;
    ensure!(!index.meshes.is_empty(), "snapshot has no saved meshes");
    let mut exact_tiles = 0usize;
    let mut lod_nodes = 0usize;
    let mut triangles = 0usize;
    for (mesh_index, mesh) in index.meshes.iter().enumerate() {
        let mut mesh_triangles = 0usize;
        for tile_index in 0..mesh.tiles.len() {
            let tile = storage::load_render_tile(run_id, mesh_index, tile_index)
                .map_err(|error| anyhow!(error))?
                .context("snapshot exact geometry tile is missing")?;
            mesh_triangles = mesh_triangles
                .checked_add(tile.triangles.len())
                .context("geometry triangle count overflow")?;
            exact_tiles += 1;
        }
        ensure!(
            mesh_triangles == mesh.triangle_count,
            "snapshot exact geometry inventory differs from its index"
        );
        triangles = triangles
            .checked_add(mesh_triangles)
            .context("geometry triangle count overflow")?;
        for node_index in 0..mesh.lod_nodes.len() {
            // Also retain the spatial view hierarchy needed for selective replay.
            // Drop each decoded tile before the next read; never gather a mesh.
            storage::load_render_lod_node(run_id, mesh_index, node_index)
                .map_err(|error| anyhow!(error))?
                .context("snapshot geometry LOD tile is missing")?;
            lod_nodes += 1;
        }
    }
    ensure!(triangles > 0, "snapshot geometry is empty");
    Ok(serde_json::json!({
        "meshes": index.meshes.len(), "exactTiles": exact_tiles,
        "lodNodes": lod_nodes, "triangles": triangles,
        "volumeElements": index.volume_element_count
    }))
}

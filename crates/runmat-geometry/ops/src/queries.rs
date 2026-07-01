use runmat_geometry_core::{GeometryAsset, Region};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryError {
    RegionNotFound,
}

pub fn find_region<'a>(
    asset: &'a GeometryAsset,
    region_id: &str,
) -> Result<&'a Region, QueryError> {
    asset
        .regions
        .iter()
        .find(|region| region.region_id == region_id)
        .ok_or(QueryError::RegionNotFound)
}

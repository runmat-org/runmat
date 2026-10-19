mod parse;

pub(super) use parse::for_dimensions;

pub(super) struct PartitionPlan {
    pub(super) sizes: Vec<Vec<usize>>,
    pub(super) offsets: Vec<Vec<usize>>,
    pub(super) cell_shape: Vec<usize>,
    pub(super) cell_count: usize,
}

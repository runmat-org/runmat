//! The original point orientation shared by every vectorized callback route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::math::optim::patternsearch) enum PointAxis {
    Rows,
    Columns,
}

impl PointAxis {
    pub fn from_initial_shape(shape: &[usize]) -> Self {
        if shape.first().copied().unwrap_or(1) > 1 && shape.get(1).copied().unwrap_or(1) == 1 {
            Self::Columns
        } else {
            Self::Rows
        }
    }

    pub fn argument_shape(self, variables: usize, points: usize) -> [usize; 2] {
        match self {
            Self::Rows => [points, variables],
            Self::Columns => [variables, points],
        }
    }

    pub fn result_shape(self, points: usize) -> [usize; 2] {
        self.argument_shape(1, points)
    }

    pub fn point_count(self, shape: &[usize], variables: usize) -> Option<usize> {
        let [rows, columns] = shape else {
            return None;
        };
        match self {
            Self::Rows if *columns == variables => Some(*rows),
            Self::Columns if *rows == variables => Some(*columns),
            Self::Rows | Self::Columns => None,
        }
    }

    /// Column-major offset after the caller validates extents and indices.
    pub fn offset(self, point: usize, coordinate: usize, points: usize, variables: usize) -> usize {
        match self {
            Self::Rows => point + coordinate * points,
            Self::Columns => coordinate + point * variables,
        }
    }
}

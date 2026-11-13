pub(super) fn magic_odd(n: usize) -> Result<Vec<usize>, String> {
    let size = n
        .checked_mul(n)
        .ok_or_else(|| "magic: dimension is too large".to_string())?;
    let mut data = vec![0usize; size];
    let mut row = 0usize;
    let mut col = n / 2;
    for val in 1..=size {
        data[idx(row, col, n)] = val;
        let next_row = (row + n - 1) % n;
        let next_col = (col + 1) % n;
        if data[idx(next_row, next_col, n)] != 0 {
            row = (row + 1) % n;
        } else {
            row = next_row;
            col = next_col;
        }
    }
    Ok(data)
}

pub(super) fn magic_doubly_even(n: usize) -> Result<Vec<usize>, String> {
    let size = n
        .checked_mul(n)
        .ok_or_else(|| "magic: dimension is too large".to_string())?;
    let mut data = vec![0usize; size];

    for row in 0..n {
        for col in 0..n {
            let value = row * n + col + 1;
            let row_mod = row % 4;
            let col_mod = col % 4;
            let swap = row_mod == col_mod || row_mod + col_mod == 3;
            let final_value = if swap { size + 1 - value } else { value };
            data[idx(row, col, n)] = final_value;
        }
    }

    Ok(data)
}

pub(super) fn magic_singly_even(n: usize) -> Result<Vec<usize>, String> {
    let size = n
        .checked_mul(n)
        .ok_or_else(|| "magic: dimension is too large".to_string())?;
    let m = n / 2;
    let m_sq = m
        .checked_mul(m)
        .ok_or_else(|| "magic: dimension is too large".to_string())?;
    let base = magic_odd(m)?;
    let mut data = vec![0usize; size];

    for row in 0..m {
        for col in 0..m {
            let value = base[idx(row, col, m)];
            data[idx(row, col, n)] = value;
            data[idx(row, col + m, n)] = value + 2 * m_sq;
            data[idx(row + m, col, n)] = value + 3 * m_sq;
            data[idx(row + m, col + m, n)] = value + m_sq;
        }
    }

    let k = (m - 1) / 2;

    for row in 0..m {
        if row == k {
            continue;
        }
        for col in 0..k {
            swap_cells(&mut data, n, row, col, row + m, col);
        }
    }

    if k > 0 {
        let start = m - k + 1;
        for row in 0..m {
            for col in start..m {
                swap_cells(&mut data, n, row, col + m, row + m, col + m);
            }
        }
    }

    // Center row `k`: swap columns 1..=k (instead of 0..k-1 used for other rows).
    for col in 1..=k {
        swap_cells(&mut data, n, k, col, k + m, col);
    }

    Ok(data)
}

fn idx(row: usize, col: usize, n: usize) -> usize {
    row + col * n
}

fn swap_cells(
    data: &mut [usize],
    n: usize,
    row_a: usize,
    col_a: usize,
    row_b: usize,
    col_b: usize,
) {
    let idx_a = idx(row_a, col_a, n);
    let idx_b = idx(row_b, col_b, n);
    data.swap(idx_a, idx_b);
}

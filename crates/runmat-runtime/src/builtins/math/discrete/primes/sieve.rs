pub(super) fn primes_through(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let limit = limit as usize;
    let mut composite = vec![false; limit + 1];
    let mut prime = 2usize;
    while prime <= limit / prime {
        if !composite[prime] {
            let mut multiple = prime * prime;
            while multiple <= limit {
                composite[multiple] = true;
                multiple += prime;
            }
        }
        prime += 1;
    }
    (2..=limit)
        .filter(|&candidate| !composite[candidate])
        .map(|candidate| candidate as u64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::primes_through;

    #[test]
    fn sieve_handles_empty_and_populated_ranges() {
        assert!(primes_through(1).is_empty());
        assert_eq!(primes_through(12), vec![2, 3, 5, 7, 11]);
    }
}

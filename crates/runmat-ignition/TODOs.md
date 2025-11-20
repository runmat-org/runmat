
- [ ] Branch conditions still rely on `Value::try_into<f64>()`. When a predicate returns `Value::Bool(true)` (e.g. `if isa(gpuArray, 'gpuArray')`), the VM panics with “cannot convert Bool(true) to f64”. Move truthiness checks to accept logical scalars directly (similar to MATLAB) so we can guard `gather`/`isa` properly without crashing.


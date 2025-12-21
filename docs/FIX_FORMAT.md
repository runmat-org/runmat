# MATLAB Output Format Compatibility

## Current RunMat Behavior

```
runmat> 1+1
ans = 2
```

## MATLAB Default Behavior

```matlab
>> 1+1

ans =

     2
```

## MATLAB with `format compact`

```matlab
>> format compact
>> 1+1
ans =
     2
```

---

## Required Changes

### 1. Output Structure
- `ans =` should be on its own line
- Value should be on the next line, indented (5 spaces)

### 2. Format Modes
| Mode | Description |
|------|-------------|
| Default | Blank line before `ans =`, blank line after `ans =`, then value |
| `format compact` | No extra blank lines |

### 3. Implementation Location

Primary change in: `crates/runmat-repl/src/main.rs` around line 120:

```rust
// Current:
println!("ans = {value}");

// Should become (default format):
println!();
println!("ans =");
println!();
println!("     {value}");

// Or (format compact):
println!("ans =");
println!("     {value}");
```

### 4. State Tracking Needed
- Add `format_compact: bool` to `ReplEngine` struct
- Implement `format` command to toggle modes
- Apply formatting in result display logic

---

## Additional Format Options (Future)

MATLAB supports many format variants:
- `format short` (default, 5 digits)
- `format long` (15 digits)
- `format shorte` / `format longe` (scientific notation)
- `format shortg` / `format longg` (best of fixed or scientific)
- `format bank` (2 decimal places)
- `format hex` / `format rat`

These affect numeric precision display, separate from `compact` which only affects spacing.

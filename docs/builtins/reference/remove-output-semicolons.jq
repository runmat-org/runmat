# Remove trailing semicolon from each example's "output" field.
# Usage:
#   jq -f remove-output-semicolons.jq abs.json
#   for f in builtins-json/*.json; do jq -f remove-output-semicolons.jq "$f" > "$f.tmp" && mv "$f.tmp" "$f"; done
.examples |= map(
  if has("output") and (.output | type) == "string" then
    .output |= (if endswith(";") then .[0:-1] else . end)
  else
    .
  end
)

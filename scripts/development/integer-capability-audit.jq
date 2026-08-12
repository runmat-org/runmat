def has_capabilities:
  ((.integer_capabilities // []) | length) > 0;

def has_audit:
  .integer_audit != null;

def is_settled:
  has_capabilities or has_audit;

def screened_input_types:
  [
    .signatures[].inputs[].ty
    | select(
        . == "Any"
        or . == "NumericArray"
        or . == "NumericScalar"
        or . == "IntegerScalar"
        or . == "SizeArg"
        or . == "LikePrototype"
      )
  ]
  | unique;

def is_screen_candidate:
  (screened_input_types | length) > 0;

def queue_records:
  [
    .builtins
    | to_entries[]
    | select(.value | is_screen_candidate and (is_settled | not))
    | .value.name as $name
    | {
        catalog_position: (.key + 1),
        name: $name,
        signature_count: (.value.signatures | length),
        screened_input_types: (.value | screened_input_types),
        signature_labels: [.value.signatures[].label],
        reference: (if ($reference_names | index($name)) != null then "docs/builtins/reference/" + $name + ".json" else null end)
      }
  ]
  | to_entries
  | map(. as $entry | $entry.value + {queue_position: ($entry.key + 1)});

def summary_rows:
  .builtins as $builtins
  | [
      ["metric", "count", "interpretation"],
      ["public_descriptor_records", ($builtins | length), "All checked-in public builtin descriptor records."],
      ["integer_capability_builtin_names", ($builtins | map(select(has_capabilities)) | length), "Builtin names with one or more settled per-form integer capability records."],
      ["integer_capability_forms", ($builtins | map(.integer_capabilities // []) | flatten | length), "Settled call-form records across capability-bearing builtin names."],
      ["integer_alias_builtin_names", ($builtins | map(select(.integer_audit.kind == "AliasOf")) | length), "Builtin names whose integer contract is explicitly inherited from a capability-bearing canonical builtin."],
      ["integer_inapplicable_builtin_names", ($builtins | map(select(.integer_audit.kind == "NotApplicable")) | length), "Audited builtin names with no integer data, control, class-preserving output, or backend surface."],
      ["signature_screen_candidates", ($builtins | map(select(is_screen_candidate)) | length), "Conservative triage population selected by numeric, Any, size, or prototype input descriptors."],
      ["signature_screen_settled", ($builtins | map(select(is_screen_candidate and is_settled)) | length), "Screen-positive builtin names carrying capability records or an explicit alias/inapplicable audit disposition."],
      ["signature_screen_untriaged", ($builtins | map(select(is_screen_candidate and (is_settled | not))) | length), "Screen-positive audit queue; this is an upper bound, not a count of missing integer overloads or defects."],
      ["signature_screen_excluded", ($builtins | map(select(is_screen_candidate | not)) | length), "Descriptors with no screened numeric, Any, size, or prototype input."]
    ];

def filtered_queue:
  queue_records
  | map(select(($name_regex == "") or (.name | test($name_regex))))
  | map(select(($input_type == "") or (.screened_input_types | index($input_type) != null)));

if $command == "summary" then
  summary_rows
elif $command == "queue" then
  filtered_queue[$offset:($offset + $limit)]
elif $command == "packet" then
  queue_records as $queue
  | [.builtins[].name] as $all_names
  | ($requested_names | unique) as $unique_names
  | {
      queue_total: ($queue | length),
      requested_count: ($requested_names | length),
      selected_count: ($unique_names | length),
      recommended_size: {minimum: 8, maximum: 25},
      errors:
        ([if ($requested_names | length) != ($unique_names | length) then "cohort contains duplicate names" else empty end]
        + [if ($unique_names | length) < 8 or ($unique_names | length) > 25 then "cohort must contain 8-25 unique names" else empty end]
        + [$unique_names[] as $name | if ($all_names | index($name)) == null then "unknown builtin: " + $name elif ($queue | map(.name) | index($name)) == null then "builtin is not currently untriaged: " + $name else empty end]),
      records: [$requested_names[] as $name | $queue[] | select(.name == $name)]
    }
else
  error("unknown command: " + $command)
end

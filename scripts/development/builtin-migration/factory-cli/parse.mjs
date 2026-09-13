import {
  COMMANDS, newCommandOptions, OPTION_FIELDS, validateCommandOptions,
} from "./contract.mjs";

export function parseFactoryCliArguments(arguments_) {
  if (arguments_.includes("--help") || arguments_.includes("-h")) return { help: true };
  const remaining = [...arguments_];
  const command = remaining.shift();
  if (!COMMANDS.includes(command)) throw new Error(`expected ${COMMANDS.join(", ")}; use --help`);
  const options = newCommandOptions(command);
  const suppliedOptions = [];
  if (command === "prepare") options.identity = requireValue(remaining, "prepare identity");
  while (remaining.length) {
    const option = remaining.shift();
    const field = OPTION_FIELDS[option];
    if (!field) throw new Error(`unknown option ${option}`);
    suppliedOptions.push(option);
    options[field] = requireValue(remaining, option);
  }
  validateCommandOptions(options, suppliedOptions);
  return options;
}

function requireValue(arguments_, option) {
  const value = arguments_.shift();
  if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`);
  return value;
}

module.exports = {
  extends: ["@commitlint/config-conventional"],
  // Legacy commit already present on dev; keep this exact exception only.
  ignores: [
    (message) =>
      message.trim() === "Add read-only as-of demand allocation backtest",
  ],
  rules: {
    "type-enum": [
      2,
      "always",
      ["feat", "fix", "docs", "refactor", "test", "chore", "style", "perf"],
    ],
    "subject-empty": [2, "never"],
    "type-empty": [2, "never"],
    "header-max-length": [2, "always", 200],
    "body-max-line-length": [0, "always", 200],
  },
};

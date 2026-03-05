/**
 * Configuration helpers for the LangSmith tracer extension.
 *
 * Reads LANGSMITH_* env vars and merges with optional plugin config.
 * Keeps all LangSmith-specific setup out of index.ts.
 */

import { Client } from "langsmith";

export type TracerConfig = {
  apiKey: string;
  project: string;
  endpoint: string;
};

const DEFAULT_PROJECT = "openclaw-agent-runs";
const DEFAULT_ENDPOINT = "https://api.smith.langchain.com";

/**
 * Returns true if tracing should be enabled:
 *   - LANGSMITH_API_KEY env var is set, OR
 *   - pluginConfig.apiKey is provided
 */
export function isEnabled(pluginCfg?: Record<string, unknown>): boolean {
  if (process.env["LANGSMITH_API_KEY"]) return true;
  if (typeof pluginCfg?.["apiKey"] === "string" && pluginCfg["apiKey"]) return true;
  return false;
}

/**
 * Resolve final config from env vars (preferred) and plugin config (fallback).
 */
export function resolveConfig(pluginCfg?: Record<string, unknown>): TracerConfig {
  const apiKey =
    process.env["LANGSMITH_API_KEY"] ??
    (typeof pluginCfg?.["apiKey"] === "string" ? pluginCfg["apiKey"] : "") ??
    "";

  const project =
    process.env["LANGSMITH_PROJECT"] ??
    (typeof pluginCfg?.["project"] === "string" ? pluginCfg["project"] : undefined) ??
    DEFAULT_PROJECT;

  const endpoint =
    process.env["LANGSMITH_ENDPOINT"] ??
    (typeof pluginCfg?.["endpoint"] === "string" ? pluginCfg["endpoint"] : undefined) ??
    DEFAULT_ENDPOINT;

  return { apiKey, project, endpoint };
}

/**
 * Build a LangSmith Client from resolved config.
 */
export function buildClient(cfg: TracerConfig): Client {
  return new Client({
    apiKey: cfg.apiKey,
    apiUrl: cfg.endpoint,
  });
}

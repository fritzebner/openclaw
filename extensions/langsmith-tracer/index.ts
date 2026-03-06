/**
 * LangSmith Tracer — OpenClaw fork extension.
 *
 * Traces every agent turn to LangSmith using the OpenClaw plugin hook system.
 * No core files are modified; this extension is entirely additive.
 *
 * ## Quick enable
 *
 *   export LANGSMITH_API_KEY=ls__...
 *   export LANGSMITH_PROJECT=openclaw          # optional, default: "default"
 *   export LANGSMITH_TRACING_V2=true           # required by LangSmith
 *
 * Then add to your openclaw.yml:
 *
 *   plugins:
 *     entries:
 *       langsmith-tracer:
 *         enabled: true
 *
 * ## Architecture
 *
 * See extensions/langsmith-tracer/src/tracer.ts for the RunTree state machine
 * and extensions/langsmith-tracer/README.md for the full reference.
 *
 * ## Hook strategy
 *
 * Session creation uses `before_prompt_build` — fires exactly once per attempt
 * with `{ prompt, messages }`, independently of other plugins' return values.
 *
 * We deliberately avoid `before_agent_start` for session creation: it fires twice
 * per turn (model-resolve phase without messages, then prompt-build phase with
 * messages), and the second fire is skipped if any other plugin already returned
 * a result from the first fire (e.g. user-context-inject returning prependContext).
 *
 * Primary LLM tracking uses `sdk_llm_start` / `sdk_llm_end` hooks — these fire
 * for EACH LLM API call within the SDK's tool loop, giving full visibility into
 * multi-step turns (LLM1 → tool → LLM2 → response).
 *
 * `llm_input` / `llm_output` are kept as fallback for non-SDK code paths.
 *
 * ## Correlation key
 *
 * Agent hooks carry PluginHookAgentContext with agentId.
 * Tool hooks carry PluginHookToolContext with agentId.
 * We use agentId as the primary key so all events correlate.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { buildClient, isEnabled, resolveConfig } from "./src/config.js";
import { LangSmithTracer } from "./src/tracer.js";

export default {
  id: "langsmith-tracer",
  name: "LangSmith Tracer",
  description:
    "Traces OpenClaw agent runs (LLM calls, tool calls, token usage) to LangSmith. " +
    "Enable by setting LANGSMITH_API_KEY. No-op when the key is absent.",

  register(api: OpenClawPluginApi) {
    const pluginCfg = api.pluginConfig as Record<string, unknown> | undefined;

    if (!isEnabled(pluginCfg)) {
      api.logger.info(
        "langsmith-tracer: LANGSMITH_API_KEY not set — tracing disabled (set the env var to enable)",
      );
      return;
    }

    const cfg = resolveConfig(pluginCfg);
    const client = buildClient(cfg);
    const tracer = new LangSmithTracer({
      client,
      projectName: cfg.project,
      logger: api.logger,
    });

    api.logger.info(
      `langsmith-tracer: tracing enabled → project "${cfg.project}" at ${cfg.endpoint}`,
    );

    // ── before_prompt_build ─────────────────────────────────────────────────
    // Creates the root "chain" run for the full agent turn.
    // We use before_prompt_build (not before_agent_start) because it fires
    // exactly once per attempt with messages, independently of other plugins'
    // return values. before_agent_start's second fire (with messages) is
    // skipped if any plugin already returned a result in the first fire.
    api.on("before_prompt_build", async (event, ctx) => {
      const key = ctx.agentId ?? ctx.sessionId ?? "unknown";
      await tracer.onAgentStart(key, event);
    });

    // ── sdk_llm_start ───────────────────────────────────────────────────────
    // Creates a child "llm" run for EACH LLM API call within the tool loop.
    api.on("sdk_llm_start", async (event, ctx) => {
      const key = ctx.agentId ?? "unknown";
      await tracer.onSdkLlmStart(key, event);
    });

    // ── sdk_llm_end ─────────────────────────────────────────────────────────
    // Closes the LLM run with per-call usage data.
    api.on("sdk_llm_end", async (event, ctx) => {
      const key = ctx.agentId ?? "unknown";
      await tracer.onSdkLlmEnd(key, event);
    });

    // ── llm_input (fallback) ────────────────────────────────────────────────
    // Fires once per attempt() — used only if sdk_llm_start hasn't fired.
    api.on("llm_input", async (event, ctx) => {
      const key = ctx.agentId ?? event.sessionId ?? "unknown";
      await tracer.onLlmInput(key, event);
    });

    // ── before_tool_call ────────────────────────────────────────────────────
    // Creates a child "tool" run under the root chain.
    api.on("before_tool_call", async (event, ctx) => {
      const key = ctx.agentId ?? "unknown";
      await tracer.onBeforeToolCall(key, event);
    });

    // ── after_tool_call ─────────────────────────────────────────────────────
    // Closes the tool run with its result or error.
    api.on("after_tool_call", async (event, ctx) => {
      const key = ctx.agentId ?? "unknown";
      await tracer.onAfterToolCall(key, event);
    });

    // ── llm_output (fallback) ───────────────────────────────────────────────
    // Fires once per attempt() — used only if sdk_llm_end hasn't fired.
    api.on("llm_output", async (event, ctx) => {
      const key = ctx.agentId ?? event.sessionId ?? "unknown";
      await tracer.onLlmOutput(key, event);
    });

    // ── agent_end ───────────────────────────────────────────────────────────
    // Closes the root run and removes the session from the state map.
    api.on("agent_end", async (event, ctx) => {
      const key = ctx.agentId ?? ctx.sessionId ?? "unknown";
      await tracer.onAgentEnd(key, event);
    });
  },
};

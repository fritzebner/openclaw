/**
 * LangSmithTracer — RunTree state machine for the OpenClaw agent loop.
 *
 * ## Trace hierarchy
 *
 * Each agent turn (one user message → one reply) produces this structure:
 *
 *   openclaw-agent  [chain]          ← before_agent_start … agent_end
 *     anthropic/claude-…  [llm]      ← sdk_llm_start … sdk_llm_end (call 1)
 *     bash  [tool]                   ← before_tool_call … after_tool_call
 *     read_file  [tool]
 *     anthropic/claude-…  [llm]      ← sdk_llm_start … sdk_llm_end (call 2)
 *
 * Tools and LLM calls are siblings under the root chain (not nested under LLM).
 * This matches the LangSmith UI convention where tools appear as peer steps.
 *
 * ## Hook event sources
 *
 * - `sdk_llm_start` / `sdk_llm_end`: Fire from the pi-agent SDK's subscription
 *   handler for EACH LLM API call within the tool loop. These are the primary
 *   source for LLM child runs — they capture intermediate LLM calls that the
 *   outer `llm_input`/`llm_output` hooks miss.
 *
 * - `llm_input` / `llm_output`: Fire ONCE per attempt() at the outer boundary.
 *   Kept as fallback if sdk_llm_* hooks don't fire (e.g. older SDK versions).
 *
 * - `before_tool_call` / `after_tool_call`: Fire for each tool call within the
 *   SDK's tool loop.
 *
 * - `before_agent_start` / `agent_end`: Fire at the boundaries of each attempt().
 *
 * ## Deferred close (agent_end / sdk_llm_end race)
 *
 * agent_end fires BEFORE the final sdk_llm_end (both fire-and-forget). We use
 * a counter (pendingLlmCallCount) and a grace-period timer:
 *   - incremented by onSdkLlmStart, decremented by onSdkLlmEnd
 *   - agent_end sets agentEnded; if count > 0, onSdkLlmEnd will finalize
 *   - if count == 0, a grace-period timer fires _forceClose
 *
 * ## after_tool_call context missing agentId
 *
 * OpenClaw's after_tool_call hook context historically passed agentId: undefined.
 * This is now fixed (ctx.params.agentId is threaded through), but we keep the
 * fallback scan for safety.
 *
 * ## Heartbeat filtering
 *
 * Heartbeat/cron runs (prompt contains "HEARTBEAT.md" or "heartbeat") are
 * skipped to reduce noise.
 *
 * ## Error safety
 *
 * Every public method catches all errors internally — a LangSmith failure must
 * never break the agent loop. Errors are logged to the provided logger.
 */

import { RunTree } from "langsmith";
import type { Client } from "langsmith";

// Hook event types (mirrors src/plugins/types.ts, inlined to avoid core imports)
export type SdkLlmStartEvent = {
  provider?: string;
  model?: string;
};

export type SdkLlmEndEvent = {
  provider?: string;
  model?: string;
  stopReason?: string;
  usage?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
    total?: number;
  };
};

export type LlmInputEvent = {
  runId: string;
  sessionId: string;
  provider: string;
  model: string;
  systemPrompt?: string;
  prompt: string;
  historyMessages: unknown[];
  imagesCount: number;
};

export type LlmOutputEvent = {
  runId: string;
  sessionId: string;
  provider: string;
  model: string;
  assistantTexts: string[];
  lastAssistant?: unknown;
  usage?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
    total?: number;
  };
};

export type AgentStartEvent = {
  prompt: string;
  messages?: unknown[];
};

export type AgentEndEvent = {
  messages: unknown[];
  success: boolean;
  error?: string;
  durationMs?: number;
};

export type BeforeToolCallEvent = {
  toolName: string;
  params: Record<string, unknown>;
};

export type AfterToolCallEvent = {
  toolName: string;
  params: Record<string, unknown>;
  result?: unknown;
  error?: string;
  durationMs?: number;
};

type PendingTool = {
  /** Tool name — used to match after_tool_call when agentId is absent. */
  name: string;
  run: RunNode;
};

type SessionTrace = {
  rootRun: RunNode;
  /** The LLM call currently in progress; null between turns. */
  currentLlmRun: RunNode | null;
  /**
   * Stack of pending tool runs. Keyed by toolName so after_tool_call can
   * match by name even when the agentId is absent from the hook context.
   */
  pendingToolStack: PendingTool[];
  /**
   * Number of LLM calls currently in flight (incremented by onSdkLlmStart,
   * decremented by onSdkLlmEnd). Finalization only happens when this reaches
   * zero AND agentEnded is set.
   */
  pendingLlmCallCount: number;
  /**
   * Set by agent_end. Triggers finalization once pendingLlmCallCount hits 0.
   */
  agentEnded?: { success: boolean; error?: string };
  /**
   * Grace-period timer started by agent_end when pendingLlmCallCount === 0.
   * Cancelled if a new sdk_llm_start arrives before the timer fires.
   */
  forceCloseTimer?: ReturnType<typeof setTimeout>;
  /** True if sdk_llm_start has fired at least once (vs legacy llm_input fallback). */
  hasSdkLlmEvents: boolean;
};

export type TracerLogger = {
  warn: (msg: string) => void;
  info?: (msg: string) => void;
};

/** Minimal interface for a run node — matches RunTree's used methods. */
export interface RunNode {
  postRun(): Promise<void>;
  createChild(cfg: Record<string, unknown>): RunNode;
  end(outputs?: Record<string, unknown>, error?: string): Promise<void>;
  patchRun(): Promise<void>;
}

/** Factory that creates root RunNodes. Injected so tests can supply fakes. */
export type RunNodeFactory = (cfg: Record<string, unknown>) => RunNode;

/** Grace period (ms) to wait for sdk_llm_end after agent_end fires. */
const CLOSE_GRACE_MS = 3000;

/** Patterns in the prompt that indicate a heartbeat/cron run, not a real user turn. */
const HEARTBEAT_PATTERNS = ["HEARTBEAT.md", "heartbeat"];

export class LangSmithTracer {
  private readonly projectName: string;
  private readonly logger: TracerLogger;
  private readonly createRootRun: RunNodeFactory;
  /** Active traces indexed by agentId. */
  private readonly sessions = new Map<string, SessionTrace>();

  constructor(opts: {
    client: Client;
    projectName: string;
    logger: TracerLogger;
    /** Override for testing — defaults to `new RunTree(cfg)`. */
    _runNodeFactory?: RunNodeFactory;
  }) {
    this.projectName = opts.projectName;
    this.logger = opts.logger;
    this.createRootRun = opts._runNodeFactory
      ? opts._runNodeFactory
      : (cfg) => {
          const rtCfg = cfg as Record<string, unknown>;
          return new RunTree({
            name: rtCfg.name as string,
            run_type: rtCfg.run_type as string,
            project_name: rtCfg.project_name as string,
            inputs: rtCfg.inputs as Record<string, unknown>,
            extra: rtCfg.extra as Record<string, unknown>,
            client: opts.client,
          });
        };
  }

  // ── before_agent_start ────────────────────────────────────────────────────

  async onAgentStart(sessionId: string, event: AgentStartEvent): Promise<void> {
    try {
      // Skip heartbeat/cron runs to reduce noise.
      if (this._isHeartbeat(event.prompt)) {
        return;
      }

      // OpenClaw fires before_agent_start TWICE per turn:
      //   1. Model-resolve phase (event.messages === undefined) — skip
      //   2. Prompt-build phase (event.messages is an array) — create root run
      if (event.messages === undefined) {
        return;
      }

      if (this.sessions.has(sessionId)) {
        const existing = this.sessions.get(sessionId)!;
        if (existing.agentEnded) {
          // Previous iteration completed. Reuse root run for multi-step turns.
          if (existing.forceCloseTimer) {
            clearTimeout(existing.forceCloseTimer);
            existing.forceCloseTimer = undefined;
          }
          existing.agentEnded = undefined;
          existing.currentLlmRun = null;
          existing.pendingLlmCallCount = 0;
          existing.pendingToolStack = [];
          return; // Reuse existing root run
        }
        // Session still active — force-close stale trace.
        this.logger.warn(
          `langsmith-tracer: session ${sessionId} still active — closing stale trace`,
        );
        await this._forceCloseSession(sessionId, false, "replaced by new agent start");
      }

      const rootRun = this.createRootRun({
        name: "openclaw-agent",
        run_type: "chain",
        project_name: this.projectName,
        inputs: { prompt: event.prompt },
        extra: {
          metadata: { sessionId },
        },
      });
      await rootRun.postRun();

      this.sessions.set(sessionId, {
        rootRun,
        currentLlmRun: null,
        pendingToolStack: [],
        pendingLlmCallCount: 0,
        hasSdkLlmEvents: false,
      });
    } catch (err) {
      this.logger.warn(`langsmith-tracer: onAgentStart failed: ${String(err)}`);
    }
  }

  // ── sdk_llm_start ──────────────────────────────────────────────────────────
  // Fires for EACH LLM API call within the SDK's tool loop.

  async onSdkLlmStart(sessionId: string, event: SdkLlmStartEvent): Promise<void> {
    try {
      const session = this.sessions.get(sessionId);
      if (!session) return;

      session.hasSdkLlmEvents = true;

      // Cancel grace-period timer if agent_end fired between tool call and this LLM call.
      if (session.forceCloseTimer) {
        clearTimeout(session.forceCloseTimer);
        session.forceCloseTimer = undefined;
      }

      session.pendingLlmCallCount++;

      const llmRun = session.rootRun.createChild({
        name: `${event.provider ?? "unknown"}/${event.model ?? "unknown"}`,
        run_type: "llm",
        inputs: {},
      });
      await llmRun.postRun();

      session.currentLlmRun = llmRun;
    } catch (err) {
      this.logger.warn(`langsmith-tracer: onSdkLlmStart failed: ${String(err)}`);
    }
  }

  // ── sdk_llm_end ────────────────────────────────────────────────────────────
  // Fires when each SDK LLM call completes, with per-call usage data.

  async onSdkLlmEnd(sessionId: string, event: SdkLlmEndEvent): Promise<void> {
    try {
      const session = this.sessions.get(sessionId);
      if (!session?.currentLlmRun) return;

      const llmRun = session.currentLlmRun;
      session.currentLlmRun = null;

      await llmRun.end({
        prompt_tokens: event.usage?.input ?? 0,
        completion_tokens: event.usage?.output ?? 0,
        total_tokens: event.usage?.total ?? 0,
        usage: event.usage ?? {},
        stop_reason: event.stopReason,
      });
      await llmRun.patchRun();

      session.pendingLlmCallCount = Math.max(0, session.pendingLlmCallCount - 1);

      // If agent_end has fired and all LLM calls are done, start finalization timer.
      // We use a timer (not immediate finalization) so that onAgentStart can cancel
      // and reuse the session if attempt.ts fires again for a multi-step turn.
      if (session.agentEnded && session.pendingLlmCallCount === 0 && !session.forceCloseTimer) {
        const { success, error } = session.agentEnded;
        session.forceCloseTimer = setTimeout(() => {
          this._forceCloseSession(sessionId, success, error).catch((err) => {
            this.logger.warn(`langsmith-tracer: deferred close failed: ${String(err)}`);
          });
        }, CLOSE_GRACE_MS);
      }
    } catch (err) {
      this.logger.warn(`langsmith-tracer: onSdkLlmEnd failed: ${String(err)}`);
    }
  }

  // ── llm_input (fallback) ───────────────────────────────────────────────────
  // Fires once per attempt() at the outer boundary. Used only if sdk_llm_start
  // hasn't fired (e.g. older SDK or non-embedded runs).

  async onLlmInput(sessionId: string, event: LlmInputEvent): Promise<void> {
    try {
      const session = this.sessions.get(sessionId);
      if (!session) return;

      // If sdk_llm_start already fired, skip — sdk events are authoritative.
      if (session.hasSdkLlmEvents) return;

      // Cancel grace-period timer (same logic as onSdkLlmStart).
      if (session.forceCloseTimer) {
        clearTimeout(session.forceCloseTimer);
        session.forceCloseTimer = undefined;
      }

      session.pendingLlmCallCount++;

      const llmRun = session.rootRun.createChild({
        name: `${event.provider}/${event.model}`,
        run_type: "llm",
        inputs: {
          system: event.systemPrompt ?? "",
          messages: event.historyMessages,
          images_count: event.imagesCount,
        },
      });
      await llmRun.postRun();

      session.currentLlmRun = llmRun;
    } catch (err) {
      this.logger.warn(`langsmith-tracer: onLlmInput failed: ${String(err)}`);
    }
  }

  // ── before_tool_call ──────────────────────────────────────────────────────

  async onBeforeToolCall(sessionId: string, event: BeforeToolCallEvent): Promise<void> {
    try {
      const session = this.sessions.get(sessionId);
      if (!session) return;

      // Tools are children of the root chain (peer to LLM runs).
      const toolRun = session.rootRun.createChild({
        name: event.toolName,
        run_type: "tool",
        inputs: event.params,
      });
      await toolRun.postRun();

      session.pendingToolStack.push({ name: event.toolName, run: toolRun });
    } catch (err) {
      this.logger.warn(`langsmith-tracer: onBeforeToolCall failed: ${String(err)}`);
    }
  }

  // ── after_tool_call ───────────────────────────────────────────────────────

  async onAfterToolCall(sessionId: string, event: AfterToolCallEvent): Promise<void> {
    try {
      // Primary lookup by session key (agentId).
      let session = this.sessions.get(sessionId);

      // Fallback: scan all active sessions for one with a pending tool match.
      if (!session) {
        for (const s of this.sessions.values()) {
          if (s.pendingToolStack.some((t) => t.name === event.toolName)) {
            session = s;
            break;
          }
        }
      }

      if (!session) return;

      // Pop the most recent tool with this name.
      const idx = this._findLastTool(session.pendingToolStack, event.toolName);
      if (idx === -1) return;

      const [{ run: toolRun }] = session.pendingToolStack.splice(idx, 1);
      await toolRun.end(event.error ? undefined : { output: event.result }, event.error);
      await toolRun.patchRun();
    } catch (err) {
      this.logger.warn(`langsmith-tracer: onAfterToolCall failed: ${String(err)}`);
    }
  }

  // ── llm_output (fallback) ──────────────────────────────────────────────────
  // Fires once per attempt(). Used only if sdk_llm_end hasn't fired.

  async onLlmOutput(sessionId: string, event: LlmOutputEvent): Promise<void> {
    try {
      const session = this.sessions.get(sessionId);
      if (!session) return;

      // If sdk_llm_end handles this, skip.
      if (session.hasSdkLlmEvents) return;

      if (!session.currentLlmRun) return;

      const llmRun = session.currentLlmRun;
      session.currentLlmRun = null;

      await llmRun.end({
        generations: event.assistantTexts,
        prompt_tokens: event.usage?.input ?? 0,
        completion_tokens: event.usage?.output ?? 0,
        total_tokens: event.usage?.total ?? 0,
        usage: event.usage ?? {},
      });
      await llmRun.patchRun();

      session.pendingLlmCallCount = Math.max(0, session.pendingLlmCallCount - 1);

      if (session.agentEnded && session.pendingLlmCallCount === 0 && !session.forceCloseTimer) {
        const { success, error } = session.agentEnded;
        session.forceCloseTimer = setTimeout(() => {
          this._forceCloseSession(sessionId, success, error).catch((err) => {
            this.logger.warn(`langsmith-tracer: deferred close failed: ${String(err)}`);
          });
        }, CLOSE_GRACE_MS);
      }
    } catch (err) {
      this.logger.warn(`langsmith-tracer: onLlmOutput failed: ${String(err)}`);
    }
  }

  // ── agent_end ─────────────────────────────────────────────────────────────

  async onAgentEnd(sessionId: string, event: AgentEndEvent): Promise<void> {
    try {
      const session = this.sessions.get(sessionId);
      if (!session) return;

      session.agentEnded = { success: event.success, error: event.error };

      // Start a grace-period timer. If LLM calls are still pending, onSdkLlmEnd
      // or onLlmOutput will handle finalization when count reaches 0.
      // If no LLM calls pending, the timer acts as a safety net for multi-step
      // turns (another before_agent_start may arrive and cancel the timer).
      if (!session.forceCloseTimer) {
        session.forceCloseTimer = setTimeout(() => {
          this._forceCloseSession(sessionId, event.success, event.error).catch((err) => {
            this.logger.warn(`langsmith-tracer: deferred close failed: ${String(err)}`);
          });
        }, CLOSE_GRACE_MS);
      }
    } catch (err) {
      this.logger.warn(`langsmith-tracer: onAgentEnd failed: ${String(err)}`);
    }
  }

  // ── internals ─────────────────────────────────────────────────────────────

  /** Force-close a session, marking any open LLM/tool runs as errors. */
  private async _forceCloseSession(
    sessionId: string,
    success: boolean,
    error?: string,
  ): Promise<void> {
    try {
      const session = this.sessions.get(sessionId);
      if (!session) return;

      if (session.forceCloseTimer) {
        clearTimeout(session.forceCloseTimer);
        session.forceCloseTimer = undefined;
      }

      for (const { run: toolRun } of [...session.pendingToolStack].reverse()) {
        try {
          await toolRun.end(undefined, "agent ended with tool still pending");
          await toolRun.patchRun();
        } catch {
          // best-effort
        }
      }
      session.pendingToolStack = [];

      if (session.currentLlmRun) {
        try {
          await session.currentLlmRun.end(undefined, "agent ended with LLM call still pending");
          await session.currentLlmRun.patchRun();
        } catch {
          // best-effort
        }
        session.currentLlmRun = null;
      }

      await this._finalizeSession(sessionId, session, success, error);
    } catch (err) {
      this.logger.warn(`langsmith-tracer: _forceCloseSession failed: ${String(err)}`);
      this.sessions.delete(sessionId);
    }
  }

  /** Close the root run and remove the session from the map. */
  private async _finalizeSession(
    sessionId: string,
    session: SessionTrace,
    success: boolean,
    error?: string,
  ): Promise<void> {
    this.sessions.delete(sessionId);

    // Safety net: close any tools that slipped through.
    for (const { run: toolRun } of [...session.pendingToolStack].reverse()) {
      try {
        await toolRun.end(undefined, "agent ended with tool still pending");
        await toolRun.patchRun();
      } catch {
        // best-effort
      }
    }

    await session.rootRun.end(
      success ? { success: true } : undefined,
      success ? undefined : (error ?? "agent run failed"),
    );
    await session.rootRun.patchRun();
  }

  /** Find the last tool in the stack with the given name. */
  private _findLastTool(stack: PendingTool[], name: string): number {
    for (let i = stack.length - 1; i >= 0; i--) {
      if (stack[i].name === name) return i;
    }
    return -1;
  }

  /** Returns true if the prompt looks like a heartbeat/cron run. */
  private _isHeartbeat(prompt: string): boolean {
    return HEARTBEAT_PATTERNS.some((p) => prompt.includes(p));
  }

  /** Returns the number of active sessions (for testing). */
  get activeSessionCount(): number {
    return this.sessions.size;
  }
}

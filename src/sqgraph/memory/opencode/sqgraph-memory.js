// sqgraph-memory plugin for OpenCode: brings the user's long-term memory into sessions.
// Written by `sqgraph-memory setup` (rewritten on every setup, removed by `setup --uninstall`): do not edit.
//
// - Once per session, the session-start context (the user's current facts, facts about the project, what is coming up)
//   is added to the system prompt. It does not change within a session, so provider prompt caching keeps working.
// - For every user message, memories relevant to it are attached to that message as a synthetic text part (stored with
//   the message, so the conversation history stays stable).
// Both run `sqgraph-memory hook <event> --agent opencode`, which never fails: on any problem nothing is added.

import { spawn } from "node:child_process"

const COMMAND = __SQGRAPH_COMMAND__
const ENV = __SQGRAPH_ENV__
const TIMEOUT_MS = 20000

function runHook(event, payload) {
  return new Promise((resolve) => {
    let out = ""
    let settled = false
    const finish = (value) => {
      if (!settled) {
        settled = true
        resolve(value)
      }
    }
    let child
    try {
      child = spawn(COMMAND[0], [...COMMAND.slice(1), "hook", event, "--agent", "opencode"], {
        stdio: ["pipe", "pipe", "ignore"],
        env: { ...process.env, ...ENV },
        windowsHide: true,
      })
    } catch {
      return finish(null)
    }
    const timer = setTimeout(() => {
      try {
        child.kill()
      } catch {}
      finish(null)
    }, TIMEOUT_MS)
    child.on("error", () => {
      clearTimeout(timer)
      finish(null)
    })
    child.stdout.on("data", (chunk) => {
      out += chunk
    })
    child.on("close", () => {
      clearTimeout(timer)
      try {
        finish(JSON.parse(out).hookSpecificOutput.additionalContext || null)
      } catch {
        finish(null)
      }
    })
    child.stdin.on("error", () => {})
    child.stdin.end(JSON.stringify(payload))
  })
}

export const SqgraphMemory = async ({ directory }) => {
  const sessions = new Map() // sessionID -> Promise<string | null> of the session-start context

  const sessionStart = (sessionID) => {
    if (!sessions.has(sessionID)) {
      sessions.set(sessionID, runHook("session-start", { session_id: sessionID, cwd: directory, source: "startup" }))
    }
    return sessions.get(sessionID)
  }

  return {
    "experimental.chat.system.transform": async (input, output) => {
      if (!input.sessionID) return // title / summary generation: no memory
      const text = await sessionStart(input.sessionID)
      if (!text) return
      // append to the last system text: another entry would become a second system message, which some chat
      // templates (e.g. Qwen on vLLM) reject
      if (output.system.length) output.system[output.system.length - 1] += `\n\n${text}`
      else output.system.push(text)
    },

    "chat.message": async (input, output) => {
      const texts = output.parts.filter((p) => p.type === "text" && !p.synthetic)
      if (!texts.length) return
      sessionStart(input.sessionID) // start it in parallel with the prompt search
      const text = await runHook("user-prompt", {
        session_id: input.sessionID,
        cwd: directory,
        prompt: texts.map((p) => p.text).join("\n"),
      })
      if (!text) return
      const last = texts[texts.length - 1]
      output.parts.push({
        id: `${last.id}sqgmem`, // sorts right after the user's text
        sessionID: last.sessionID,
        messageID: last.messageID,
        type: "text",
        text,
        synthetic: true,
      })
    },
  }
}

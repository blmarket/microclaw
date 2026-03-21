use async_trait::async_trait;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;
use tracing::debug;

use std::collections::HashSet;
use std::process::Stdio;

use crate::codex_auth::is_codex_app_provider;
use crate::config::Config;
#[cfg(test)]
use crate::config::WorkingDirIsolation;
use microclaw_core::error::MicroClawError;
use microclaw_core::llm_types::{
    ContentBlock, Message, MessageContent, MessagesResponse, ResponseContentBlock, ToolDefinition,
    Usage,
};

/// Remove invalid `ToolResult` blocks that cannot be matched to the most recent
/// assistant `ToolUse` turn. This can happen after session compaction or
/// malformed history reconstruction.
fn sanitize_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut pending_tool_ids: HashSet<String> = HashSet::new();
    let mut sanitized = Vec::new();

    for msg in messages {
        match msg.content {
            MessageContent::Text(text) => {
                pending_tool_ids.clear();
                sanitized.push(Message {
                    role: msg.role,
                    content: MessageContent::Text(text),
                });
            }
            MessageContent::Blocks(blocks) => {
                if msg.role == "assistant" {
                    let assistant_tool_ids: HashSet<String> = blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
                            _ => None,
                        })
                        .collect();
                    pending_tool_ids = assistant_tool_ids;
                    sanitized.push(Message {
                        role: msg.role,
                        content: MessageContent::Blocks(blocks),
                    });
                    continue;
                }

                if msg.role != "user" {
                    pending_tool_ids.clear();
                    sanitized.push(Message {
                        role: msg.role,
                        content: MessageContent::Blocks(blocks),
                    });
                    continue;
                }

                let has_tool_results = blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
                if !has_tool_results {
                    pending_tool_ids.clear();
                    sanitized.push(Message {
                        role: msg.role,
                        content: MessageContent::Blocks(blocks),
                    });
                    continue;
                }

                let mut filtered = Vec::new();
                for block in blocks {
                    let keep = match &block {
                        ContentBlock::ToolResult { tool_use_id, .. } => {
                            pending_tool_ids.contains(tool_use_id)
                        }
                        _ => true,
                    };
                    if keep {
                        if let ContentBlock::ToolResult { tool_use_id, .. } = &block {
                            pending_tool_ids.remove(tool_use_id);
                        }
                        filtered.push(block);
                    }
                }

                if !filtered.is_empty() {
                    sanitized.push(Message {
                        role: msg.role,
                        content: MessageContent::Blocks(filtered),
                    });
                }
            }
        }
    }

    sanitized
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn send_message(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<MessagesResponse, MicroClawError>;

    async fn send_message_with_model(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        _model_override: Option<&str>,
    ) -> Result<MessagesResponse, MicroClawError> {
        self.send_message(system, messages, tools).await
    }

    async fn send_message_stream(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        text_tx: Option<&UnboundedSender<String>>,
    ) -> Result<MessagesResponse, MicroClawError> {
        let response = self.send_message(system, messages, tools).await?;
        if let Some(tx) = text_tx {
            for block in &response.content {
                if let ResponseContentBlock::Text { text } = block {
                    let _ = tx.send(text.clone());
                }
            }
        }
        Ok(response)
    }

    async fn send_message_stream_with_model(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        text_tx: Option<&UnboundedSender<String>>,
        _model_override: Option<&str>,
    ) -> Result<MessagesResponse, MicroClawError> {
        self.send_message_stream(system, messages, tools, text_tx)
            .await
    }
}

pub fn create_provider(config: &Config) -> Box<dyn LlmProvider> {
    if !is_codex_app_provider(&config.llm_provider) {
        debug!(
            provider = %config.llm_provider,
            "Unsupported llm_provider reached create_provider; using codex-app provider"
        );
    }
    Box::new(CodexAppServerProvider::new(config))
}

type CodexAppServerLines = tokio::io::Lines<BufReader<tokio::process::ChildStdout>>;

#[derive(Default)]
struct CodexAppServerTurnOutcome {
    turn_id: String,
    usage: Option<Usage>,
}

pub struct CodexAppServerProvider {
    model: String,
}

impl CodexAppServerProvider {
    pub fn new(config: &Config) -> Self {
        Self {
            model: config.model.clone(),
        }
    }

    async fn run_request(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        model_override: Option<&str>,
    ) -> Result<MessagesResponse, MicroClawError> {
        let sanitized_messages = sanitize_messages(messages);
        let model = model_override
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(&self.model);

        let mut child = Command::new("codex")
            .arg("app-server")
            .arg("--listen")
            .arg("stdio://")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                MicroClawError::LlmApi(format!("Failed to start `codex app-server`: {e}"))
            })?;

        let mut stdin = child.stdin.take().ok_or_else(|| {
            MicroClawError::LlmApi("`codex app-server` did not expose stdin".into())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            MicroClawError::LlmApi("`codex app-server` did not expose stdout".into())
        })?;
        let stderr = child.stderr.take();
        let stderr_task = stderr.map(|stderr| {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                let mut buf = String::new();
                while let Ok(Some(line)) = lines.next_line().await {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(trimmed);
                }
                buf
            })
        });
        let mut stdout_lines = BufReader::new(stdout).lines();

        let result = async {
            codex_app_server_write_request(
                &mut stdin,
                1,
                "initialize",
                json!({
                    "clientInfo": {
                        "name": "microclaw",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "capabilities": {
                        "experimentalApi": false,
                    }
                }),
            )
            .await?;
            let _ = codex_app_server_wait_for_response(&mut stdout_lines, &mut stdin, 1).await?;

            codex_app_server_write_request(
                &mut stdin,
                2,
                "thread/start",
                json!({
                    "model": model,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "serviceName": "microclaw",
                    "baseInstructions": codex_app_server_base_instructions(system),
                    "developerInstructions": codex_app_server_developer_instructions(),
                    "personality": "pragmatic",
                    "ephemeral": true,
                    "experimentalRawEvents": false,
                    "persistExtendedHistory": false,
                    "config": {
                        "web_search": "disabled",
                        "tools": {
                            "web_search": null,
                            "view_image": false,
                        }
                    }
                }),
            )
            .await?;
            let thread_start = codex_app_server_wait_for_response(&mut stdout_lines, &mut stdin, 2)
                .await?;
            let thread_id = thread_start
                .get("thread")
                .and_then(|thread| thread.get("id"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    MicroClawError::LlmApi(
                        "codex app-server thread/start response did not include thread id".into(),
                    )
                })?
                .to_string();

            codex_app_server_write_request(
                &mut stdin,
                3,
                "turn/start",
                json!({
                    "threadId": thread_id.clone(),
                    "input": [
                        {
                            "type": "text",
                            "text": codex_app_server_turn_input(&sanitized_messages, tools.as_deref())?,
                            "text_elements": [],
                        }
                    ],
                    "outputSchema": codex_app_server_output_schema(),
                }),
            )
            .await?;
            let turn_start =
                codex_app_server_wait_for_response(&mut stdout_lines, &mut stdin, 3).await?;
            let turn_id = turn_start
                .get("turn")
                .and_then(|turn| turn.get("id"))
                .and_then(|v| v.as_str())
                .map(str::to_string);

            let completion = codex_app_server_wait_for_turn_completion(
                &mut stdout_lines,
                &mut stdin,
                turn_id.as_deref(),
            )
            .await?;

            codex_app_server_write_request(
                &mut stdin,
                4,
                "thread/read",
                json!({
                    "threadId": thread_id,
                    "includeTurns": true,
                }),
            )
            .await?;
            let thread_read =
                codex_app_server_wait_for_response(&mut stdout_lines, &mut stdin, 4).await?;
            let raw_response =
                codex_app_server_extract_agent_message(&thread_read, &completion.turn_id)
                    .ok_or_else(|| {
                        MicroClawError::LlmApi(
                            "codex app-server completed the turn without a final agent message"
                                .into(),
                        )
                    })?;

            let mut response = parse_codex_app_server_messages_response(&raw_response)?;
            if response.usage.is_none() {
                response.usage = completion.usage;
            }
            Ok(response)
        }
        .await;

        let _ = child.kill().await;
        let _ = child.wait().await;
        let stderr_output = match stderr_task {
            Some(task) => task.await.unwrap_or_default(),
            None => String::new(),
        };

        match result {
            Ok(response) => Ok(response),
            Err(err) => {
                if stderr_output.trim().is_empty() {
                    Err(err)
                } else {
                    Err(MicroClawError::LlmApi(format!("{err}\n{stderr_output}")))
                }
            }
        }
    }
}

async fn codex_app_server_write_request(
    stdin: &mut tokio::process::ChildStdin,
    id: i64,
    method: &str,
    params: serde_json::Value,
) -> Result<(), MicroClawError> {
    let payload = serde_json::to_string(&json!({
        "id": id,
        "method": method,
        "params": params,
    }))?;
    stdin.write_all(payload.as_bytes()).await?;
    stdin.write_all(b"\n").await?;
    stdin.flush().await?;
    Ok(())
}

async fn codex_app_server_write_result(
    stdin: &mut tokio::process::ChildStdin,
    id: serde_json::Value,
    result: serde_json::Value,
) -> Result<(), MicroClawError> {
    let payload = serde_json::to_string(&json!({
        "id": id,
        "result": result,
    }))?;
    stdin.write_all(payload.as_bytes()).await?;
    stdin.write_all(b"\n").await?;
    stdin.flush().await?;
    Ok(())
}

async fn codex_app_server_write_error(
    stdin: &mut tokio::process::ChildStdin,
    id: serde_json::Value,
    message: &str,
) -> Result<(), MicroClawError> {
    let payload = serde_json::to_string(&json!({
        "id": id,
        "error": {
            "code": -32000,
            "message": message,
        }
    }))?;
    stdin.write_all(payload.as_bytes()).await?;
    stdin.write_all(b"\n").await?;
    stdin.flush().await?;
    Ok(())
}

async fn codex_app_server_read_message(
    lines: &mut CodexAppServerLines,
) -> Result<serde_json::Value, MicroClawError> {
    loop {
        let Some(line) = lines.next_line().await? else {
            return Err(MicroClawError::LlmApi(
                "`codex app-server` closed stdout before completing the request".into(),
            ));
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(value) => return Ok(value),
            Err(err) => {
                debug!(
                    error = %err,
                    line = trimmed,
                    "Skipping non-JSON line from codex app-server stdout"
                );
            }
        }
    }
}

async fn codex_app_server_respond_to_server_request(
    stdin: &mut tokio::process::ChildStdin,
    message: &serde_json::Value,
) -> Result<(), MicroClawError> {
    let Some(id) = message.get("id").cloned() else {
        return Ok(());
    };
    let Some(method) = message.get("method").and_then(|v| v.as_str()) else {
        return Ok(());
    };

    match method {
        "item/commandExecution/requestApproval" => {
            codex_app_server_write_result(stdin, id, json!({"decision": "decline"})).await
        }
        "item/fileChange/requestApproval" => {
            codex_app_server_write_result(stdin, id, json!({"decision": "decline"})).await
        }
        "item/tool/requestUserInput" => {
            codex_app_server_write_result(stdin, id, json!({"answers": {}})).await
        }
        "mcpServer/elicitation/request" => {
            codex_app_server_write_result(
                stdin,
                id,
                json!({"action": "decline", "content": null, "_meta": null}),
            )
            .await
        }
        "item/permissions/requestApproval" => {
            codex_app_server_write_result(stdin, id, json!({"permissions": {}, "scope": "turn"}))
                .await
        }
        "item/tool/call" => {
            codex_app_server_write_result(
                stdin,
                id,
                json!({
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": "Dynamic tool calls are disabled for codex-app LlmProvider requests."
                        }
                    ],
                    "success": false,
                }),
            )
            .await
        }
        "applyPatchApproval" | "execCommandApproval" => {
            codex_app_server_write_result(stdin, id, json!({"decision": "denied"})).await
        }
        _ => {
            codex_app_server_write_error(
                stdin,
                id,
                &format!("Unsupported codex app-server request: {method}"),
            )
            .await
        }
    }
}

async fn codex_app_server_wait_for_response(
    lines: &mut CodexAppServerLines,
    stdin: &mut tokio::process::ChildStdin,
    request_id: i64,
) -> Result<serde_json::Value, MicroClawError> {
    loop {
        let message = codex_app_server_read_message(lines).await?;
        let response_id = message.get("id").and_then(|v| v.as_i64());
        if response_id == Some(request_id) {
            if let Some(result) = message.get("result") {
                return Ok(result.clone());
            }
            if let Some(error) = message.get("error") {
                return Err(MicroClawError::LlmApi(format!(
                    "codex app-server {} failed: {}",
                    request_id,
                    format_codex_app_server_error(error),
                )));
            }
        }

        if message.get("id").is_some() && message.get("method").is_some() {
            codex_app_server_respond_to_server_request(stdin, &message).await?;
        }
    }
}

async fn codex_app_server_wait_for_turn_completion(
    lines: &mut CodexAppServerLines,
    stdin: &mut tokio::process::ChildStdin,
    turn_id_hint: Option<&str>,
) -> Result<CodexAppServerTurnOutcome, MicroClawError> {
    let mut outcome = CodexAppServerTurnOutcome {
        turn_id: turn_id_hint.unwrap_or_default().to_string(),
        usage: None,
    };

    loop {
        let message = codex_app_server_read_message(lines).await?;

        if message.get("id").is_some() && message.get("method").is_some() {
            codex_app_server_respond_to_server_request(stdin, &message).await?;
            continue;
        }

        let Some(method) = message.get("method").and_then(|v| v.as_str()) else {
            continue;
        };
        let params = message
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        match method {
            "thread/tokenUsage/updated" => {
                let Some(turn_id) = params.get("turnId").and_then(|v| v.as_str()) else {
                    continue;
                };
                if !outcome.turn_id.is_empty() && outcome.turn_id != turn_id {
                    continue;
                }
                outcome.turn_id = turn_id.to_string();
                let Some(last) = params.get("tokenUsage").and_then(|v| v.get("last")) else {
                    continue;
                };
                let input_tokens = last
                    .get("inputTokens")
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok());
                let output_tokens = last
                    .get("outputTokens")
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok());
                if let (Some(input_tokens), Some(output_tokens)) = (input_tokens, output_tokens) {
                    outcome.usage = Some(Usage {
                        input_tokens,
                        output_tokens,
                    });
                }
            }
            "error" => {
                let Some(turn_id) = params.get("turnId").and_then(|v| v.as_str()) else {
                    continue;
                };
                if !outcome.turn_id.is_empty() && outcome.turn_id != turn_id {
                    continue;
                }
                if params
                    .get("willRetry")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    continue;
                }
                let message = params
                    .get("error")
                    .and_then(|v| v.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("codex app-server turn failed");
                return Err(MicroClawError::LlmApi(message.to_string()));
            }
            "turn/completed" => {
                let Some(turn) = params.get("turn") else {
                    continue;
                };
                let Some(turn_id) = turn.get("id").and_then(|v| v.as_str()) else {
                    continue;
                };
                if !outcome.turn_id.is_empty() && outcome.turn_id != turn_id {
                    continue;
                }
                outcome.turn_id = turn_id.to_string();
                match turn.get("status").and_then(|v| v.as_str()) {
                    Some("completed") => return Ok(outcome),
                    Some("failed") => {
                        let message = turn
                            .get("error")
                            .and_then(|v| v.get("message"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("codex app-server turn failed");
                        return Err(MicroClawError::LlmApi(message.to_string()));
                    }
                    Some("interrupted") => {
                        return Err(MicroClawError::LlmApi(
                            "codex app-server turn was interrupted".into(),
                        ));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

fn codex_app_server_base_instructions(system: &str) -> Option<String> {
    let trimmed = system.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn codex_app_server_developer_instructions() -> &'static str {
    "You are acting as a pure LLM compatibility layer for MicroClaw. Do not execute commands, edit files, browse the web, ask for approvals, request user input, or call built-in Codex tools. Read the serialized request payload and produce only a JSON object matching the provided output schema."
}

fn codex_app_server_turn_input(
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
) -> Result<String, MicroClawError> {
    let messages_json = serde_json::to_string_pretty(messages)?;
    let tools_json = serde_json::to_string_pretty(&tools.unwrap_or(&[]))?;
    Ok(format!(
        "Return the next assistant response for this MicroClaw request.\n\n\
Semantic rules:\n\
- The response must be a JSON object matching the output schema.\n\
- Use `content` to preserve the assistant-visible response as ordered blocks.\n\
- A text block is `{{\"type\":\"text\",\"text\":\"...\"}}`.\n\
- A tool call block is `{{\"type\":\"tool_use\",\"id\":\"call_...\",\"name\":\"tool_name\",\"input\":{{...}}}}`.\n\
- Emit tool calls only for tools listed below.\n\
- If any tool call blocks are present, `stop_reason` must be `\"tool_use\"`.\n\
- Otherwise `stop_reason` must be `\"end_turn\"`.\n\
- Treat prior `tool_use` assistant blocks and later `tool_result` user blocks as previous tool executions and their outputs.\n\
- If no visible text is needed before a tool call, use an empty `content` array or only tool blocks.\n\
- Leave `usage` omitted or null.\n\
- Do not wrap the JSON in markdown fences.\n\n\
Available tools:\n{tools_json}\n\n\
Conversation history:\n{messages_json}\n"
    ))
}

fn codex_app_server_output_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "required": ["content", "stop_reason"],
        "additionalProperties": false,
        "properties": {
            "content": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["type", "text"],
                            "additionalProperties": false,
                            "properties": {
                                "type": { "const": "text" },
                                "text": { "type": "string" }
                            }
                        },
                        {
                            "type": "object",
                            "required": ["type", "id", "name", "input"],
                            "additionalProperties": false,
                            "properties": {
                                "type": { "const": "tool_use" },
                                "id": { "type": "string" },
                                "name": { "type": "string" },
                                "input": {}
                            }
                        }
                    ]
                }
            },
            "stop_reason": {
                "type": ["string", "null"],
                "enum": ["end_turn", "tool_use", "max_tokens", null]
            },
            "usage": {
                "anyOf": [
                    { "type": "null" },
                    {
                        "type": "object",
                        "required": ["input_tokens", "output_tokens"],
                        "additionalProperties": false,
                        "properties": {
                            "input_tokens": { "type": "integer", "minimum": 0 },
                            "output_tokens": { "type": "integer", "minimum": 0 }
                        }
                    }
                ]
            }
        }
    })
}

fn codex_app_server_extract_agent_message(
    thread_read_result: &serde_json::Value,
    turn_id: &str,
) -> Option<String> {
    let turns = thread_read_result
        .get("thread")
        .and_then(|thread| thread.get("turns"))
        .and_then(|v| v.as_array())?;
    let turn = turns.iter().find(|turn| {
        turn.get("id")
            .and_then(|v| v.as_str())
            .map(|id| id == turn_id)
            .unwrap_or(false)
    })?;
    let items = turn.get("items").and_then(|v| v.as_array())?;
    let mut fallback: Option<String> = None;
    for item in items {
        if item.get("type").and_then(|v| v.as_str()) != Some("agentMessage") {
            continue;
        }
        let text = item
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if item.get("phase").and_then(|v| v.as_str()) == Some("final_answer") {
            return Some(text);
        }
        fallback = Some(text);
    }
    fallback
}

fn parse_codex_app_server_messages_response(
    text: &str,
) -> Result<MessagesResponse, MicroClawError> {
    if let Ok(response) = serde_json::from_str::<MessagesResponse>(text) {
        return Ok(normalize_codex_app_server_messages_response(response));
    }
    if let Some(json_body) = extract_first_json_object(text) {
        if let Ok(response) = serde_json::from_str::<MessagesResponse>(&json_body) {
            return Ok(normalize_codex_app_server_messages_response(response));
        }
    }
    Err(MicroClawError::LlmApi(format!(
        "Failed to parse codex app-server response as MessagesResponse. Body: {text}"
    )))
}

fn normalize_codex_app_server_messages_response(
    mut response: MessagesResponse,
) -> MessagesResponse {
    if response.content.is_empty() {
        response.content.push(ResponseContentBlock::Text {
            text: String::new(),
        });
    }
    if has_tool_use_block(&response.content) {
        response.stop_reason = Some("tool_use".into());
    } else if response.stop_reason.is_none() || response.stop_reason.as_deref() == Some("tool_use")
    {
        response.stop_reason = Some("end_turn".into());
    }
    response
}

fn has_tool_use_block(content: &[ResponseContentBlock]) -> bool {
    content
        .iter()
        .any(|b| matches!(b, ResponseContentBlock::ToolUse { .. }))
}

fn extract_first_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;

    for (idx, byte) in bytes.iter().enumerate() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match byte {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match byte {
            b'"' => in_string = true,
            b'{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            b'}' => {
                if depth == 0 {
                    continue;
                }
                depth -= 1;
                if depth == 0 {
                    let start = start?;
                    return Some(text[start..=idx].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn format_codex_app_server_error(error: &serde_json::Value) -> String {
    error
        .get("message")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| error.to_string())
}

#[async_trait]
impl LlmProvider for CodexAppServerProvider {
    async fn send_message(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<MessagesResponse, MicroClawError> {
        self.run_request(system, messages, tools, None).await
    }

    async fn send_message_with_model(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        model_override: Option<&str>,
    ) -> Result<MessagesResponse, MicroClawError> {
        self.run_request(system, messages, tools, model_override)
            .await
    }

    async fn send_message_stream(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        text_tx: Option<&UnboundedSender<String>>,
    ) -> Result<MessagesResponse, MicroClawError> {
        self.send_message_stream_with_model(system, messages, tools, text_tx, None)
            .await
    }

    async fn send_message_stream_with_model(
        &self,
        system: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        text_tx: Option<&UnboundedSender<String>>,
        model_override: Option<&str>,
    ) -> Result<MessagesResponse, MicroClawError> {
        let response = self
            .run_request(system, messages, tools, model_override)
            .await?;
        if let Some(tx) = text_tx {
            for block in &response.content {
                if let ResponseContentBlock::Text { text } = block {
                    let _ = tx.send(text.clone());
                }
            }
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_create_provider_codex_app() {
        let mut config = Config::test_defaults();
        config.llm_provider = "codex-app".into();
        config.data_dir = "/tmp".into();
        config.working_dir = "/tmp".into();
        config.working_dir_isolation = WorkingDirIsolation::Shared;
        config.web_enabled = false;
        config.web_port = 3900;
        let _provider = create_provider(&config);
    }

    #[test]
    fn test_parse_codex_app_server_messages_response_direct_json() {
        let body = r#"{"content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn"}"#;
        let parsed = parse_codex_app_server_messages_response(body).unwrap();
        assert_eq!(parsed.stop_reason.as_deref(), Some("end_turn"));
        match &parsed.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "Hello"),
            _ => panic!("Expected text block"),
        }
    }

    #[test]
    fn test_parse_codex_app_server_messages_response_extracts_json_object() {
        let body = "```json\n{\"content\":[{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"echo\",\"input\":{\"value\":1}}],\"stop_reason\":\"end_turn\"}\n```";
        let parsed = parse_codex_app_server_messages_response(body).unwrap();
        assert_eq!(parsed.stop_reason.as_deref(), Some("tool_use"));
        assert!(matches!(
            parsed.content[0],
            ResponseContentBlock::ToolUse { .. }
        ));
    }

    #[test]
    fn test_sanitize_messages_removes_orphaned_tool_results() {
        let msgs = vec![
            Message {
                role: "assistant".into(),
                content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({}),
                    thought_signature: None,
                }]),
            },
            Message {
                role: "user".into(),
                content: MessageContent::Blocks(vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "t1".into(),
                        content: "ok".into(),
                        is_error: None,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "orphan".into(),
                        content: "stale".into(),
                        is_error: None,
                    },
                ]),
            },
        ];
        let sanitized = sanitize_messages(msgs);
        assert_eq!(sanitized.len(), 2);
        if let MessageContent::Blocks(blocks) = &sanitized[1].content {
            assert_eq!(blocks.len(), 1);
            if let ContentBlock::ToolResult { tool_use_id, .. } = &blocks[0] {
                assert_eq!(tool_use_id, "t1");
            } else {
                panic!("Expected ToolResult");
            }
        } else {
            panic!("Expected Blocks");
        }
    }

    #[test]
    fn test_sanitize_messages_drops_empty_user_message() {
        let msgs = vec![Message {
            role: "user".into(),
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "orphan".into(),
                content: "stale".into(),
                is_error: None,
            }]),
        }];
        let sanitized = sanitize_messages(msgs);
        assert!(sanitized.is_empty());
    }
}

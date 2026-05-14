//! Streaming execution for Regular Responses API
//!
//! This module handles streaming request execution:
//! - `execute_tool_loop_streaming` - MCP tool loop with streaming
//! - `convert_chat_stream_to_responses_stream` - Non-MCP streaming conversion
//! - Streaming accumulators for response building

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{body::Body, extract::ws::Message, response::Response};
use bytes::Bytes;
use futures_util::StreamExt;
use openai_protocol::{
    chat::{
        ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
        ChatCompletionStreamResponse,
    },
    common::{FunctionCallResponse, ToolCall, Usage, UsageInfo},
    responses::{
        ResponseContentPart, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
        ResponsesRequest, ResponsesResponse, ResponsesUsage,
    },
};
use serde_json::{json, Value};
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput};
use tokio::sync::mpsc;
use tracing::{debug, trace, warn};
use uuid::Uuid;

use super::{
    common::{
        build_next_request, convert_mcp_tools_to_chat_tools, extract_all_tool_calls_from_chat,
        prepare_chat_tools_and_choice, ExtractedToolCall, ResponsesCallContext, ToolLoopState,
    },
    conversions,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            mcp_utils::{prepare_hosted_dispatch_args, DEFAULT_MAX_ITERATIONS},
            openai_bridge::{self, ResponseFormat},
        },
        grpc::common::responses::{
            build_sse_response, persist_response_if_needed,
            streaming::{
                attach_mcp_server_label, OutputItemKind, ResponseEventSink,
                ResponseStreamEventEmitter, SseResponseEventSink, WsResponseEventSink,
            },
            ResponsesContext,
        },
    },
};

// ============================================================================
// Non-MCP Streaming Path
// ============================================================================

fn generate_item_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::now_v7().simple())
}

enum SseDataRecord {
    ChatChunk(ChatCompletionStreamResponse),
    RawJson(String),
    Done,
}

struct SseBodyParser {
    pending: String,
    offset: usize,
}

impl SseBodyParser {
    fn new() -> Self {
        Self {
            pending: String::new(),
            offset: 0,
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        let text = String::from_utf8_lossy(chunk).replace("\r\n", "\n");
        self.pending.push_str(&text);
    }

    fn next_record(&mut self) -> Option<SseDataRecord> {
        let rel_end = self.pending[self.offset..].find("\n\n")?;
        let record_end = self.offset + rel_end;
        let record = self.pending[self.offset..record_end].trim().to_string();
        self.offset = record_end + 2;

        if self.offset > self.pending.len() / 2 {
            self.pending = self.pending[self.offset..].to_string();
            self.offset = 0;
        }

        if record.is_empty() {
            return self.next_record();
        }

        if record == "data: [DONE]" {
            return Some(SseDataRecord::Done);
        }

        let Some(json_str) = record.strip_prefix("data: ") else {
            return self.next_record();
        };
        let json_str = json_str.trim();

        match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
            Ok(chunk) => Some(SseDataRecord::ChatChunk(chunk)),
            Err(_) => Some(SseDataRecord::RawJson(json_str.to_string())),
        }
    }

    fn flush(&mut self) -> Option<SseDataRecord> {
        let remaining = self.pending[self.offset..].trim();
        if remaining.is_empty() || remaining == "data: [DONE]" {
            return None;
        }
        self.offset = self.pending.len();

        let json_str = remaining.strip_prefix("data: ")?.trim();
        match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
            Ok(chunk) => Some(SseDataRecord::ChatChunk(chunk)),
            Err(_) => Some(SseDataRecord::RawJson(json_str.to_string())),
        }
    }
}

/// Convert chat streaming response to responses streaming format
///
/// This function:
/// 1. Gets chat SSE stream from pipeline
/// 2. Intercepts and parses each SSE event
/// 3. Converts ChatCompletionStreamResponse → ResponsesResponse delta
/// 4. Accumulates response state for final persistence
/// 5. Emits transformed SSE events in responses format
pub(super) async fn convert_chat_stream_to_responses_stream(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    params: ResponsesCallContext,
    original_request: &ResponsesRequest,
) -> Response {
    debug!("Converting chat SSE stream to responses SSE format");

    // Create channel for transformed SSE events
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    let sink = SseResponseEventSink::new(tx.clone());

    // Spawn background task to transform stream
    let original_request_clone = original_request.clone();
    let ctx_clone = ctx.clone();

    #[expect(
        clippy::disallowed_methods,
        reason = "streaming task is fire-and-forget; client disconnect terminates it"
    )]
    tokio::spawn(async move {
        if let Err(e) = execute_non_mcp_stream_with_sink(
            &ctx_clone,
            chat_request,
            params,
            original_request_clone,
            &sink,
        )
        .await
        {
            warn!("Error transforming SSE stream: {}", e);
            let error_event = json!({
                "error": {
                    "message": e,
                    "type": "stream_error"
                }
            });
            let _ = sink.send_raw_json(&error_event.to_string());
        }

        let _ = sink.send_done();
    });

    build_sse_response(rx)
}

pub(super) async fn execute_non_mcp_stream_with_sink(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    params: ResponsesCallContext,
    original_request: ResponsesRequest,
    sink: &impl ResponseEventSink,
) -> Result<ResponsesResponse, String> {
    let chat_response = ctx
        .pipeline
        .execute_chat(
            chat_request,
            params.headers,
            params.model_id,
            ctx.components.clone(),
            Some(params.tenant_request_meta),
        )
        .await;
    let (_parts, body) = chat_response.into_parts();

    process_and_transform_stream(
        body,
        original_request,
        ctx.response_storage.clone(),
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.request_context.clone(),
        sink,
    )
    .await
}

async fn process_and_transform_stream(
    body: Body,
    original_request: ResponsesRequest,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    request_context: Option<StorageRequestContext>,
    sink: &impl ResponseEventSink,
) -> Result<ResponsesResponse, String> {
    let response_id = format!("resp_{}", Uuid::now_v7());
    let model = original_request.model.clone();
    let created_at = chrono::Utc::now().timestamp() as u64;
    let mut accumulator = StreamingResponseAccumulator::new(
        &original_request,
        response_id.clone(),
        model.clone(),
        created_at as i64,
    );
    let mut event_emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);
    event_emitter.set_original_request(original_request.clone());

    let event = event_emitter.emit_created();
    event_emitter
        .send_event(&event, sink)
        .map_err(|_| "Failed to send response.created event".to_string())?;

    let event = event_emitter.emit_in_progress();
    event_emitter
        .send_event(&event, sink)
        .map_err(|_| "Failed to send response.in_progress event".to_string())?;

    let mut stream = body.into_data_stream();
    let mut parser = SseBodyParser::new();
    let mut upstream_error_forwarded = false;
    let mut function_call_events = BTreeMap::new();

    'stream: while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {e}"))?;
        parser.push(&chunk);

        while let Some(record) = parser.next_record() {
            match process_non_mcp_sse_data_record(
                record,
                &mut accumulator,
                &mut event_emitter,
                sink,
                &mut function_call_events,
            )? {
                SseRecordOutcome::Continue => {}
                SseRecordOutcome::StopStream => break 'stream,
                SseRecordOutcome::UpstreamError => {
                    upstream_error_forwarded = true;
                    break 'stream;
                }
            }
        }
    }

    if !upstream_error_forwarded {
        if let Some(record) = parser.flush() {
            if let SseRecordOutcome::UpstreamError = process_non_mcp_sse_data_record(
                record,
                &mut accumulator,
                &mut event_emitter,
                sink,
                &mut function_call_events,
            )? {
                upstream_error_forwarded = true;
            }
        }
    }

    if upstream_error_forwarded {
        debug!("Upstream error payload was already forwarded; skipping response.completed");
        return Ok(accumulator.finalize_with_status(ResponseStatus::Failed));
    }

    let usage_json = accumulator.usage.as_ref().map(|u| {
        let mut usage_obj = json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens
        });

        if let Some(details) = &u.completion_tokens_details {
            if let Some(reasoning_tokens) = details.reasoning_tokens {
                usage_obj["output_tokens_details"] =
                    json!({ "reasoning_tokens": reasoning_tokens });
            }
        }

        usage_obj
    });

    let terminal_status = accumulator.response_status();
    let completed_event =
        event_emitter.emit_completed_with_status(terminal_status, usage_json.as_ref());
    event_emitter.send_event(&completed_event, sink)?;

    let final_response = accumulator.finalize();
    persist_response_if_needed(
        conversation_storage,
        conversation_item_storage,
        response_storage,
        &final_response,
        &original_request,
        request_context,
    )
    .await;

    Ok(final_response)
}

enum SseRecordOutcome {
    Continue,
    StopStream,
    UpstreamError,
}

fn process_non_mcp_sse_data_record(
    record: SseDataRecord,
    accumulator: &mut StreamingResponseAccumulator,
    event_emitter: &mut ResponseStreamEventEmitter,
    sink: &impl ResponseEventSink,
    function_call_events: &mut BTreeMap<usize, FunctionCallEventState>,
) -> Result<SseRecordOutcome, String> {
    match record {
        SseDataRecord::Done => Ok(SseRecordOutcome::StopStream),
        SseDataRecord::ChatChunk(chat_chunk) => {
            accumulator.process_chunk(&chat_chunk);

            let has_tool_call_delta = chat_chunk
                .choices
                .first()
                .and_then(|choice| choice.delta.tool_calls.as_ref())
                .is_some_and(|tool_calls| !tool_calls.is_empty());
            if has_tool_call_delta || !function_call_events.is_empty() {
                process_non_mcp_function_call_chunk(
                    &chat_chunk,
                    event_emitter,
                    sink,
                    function_call_events,
                )?;
            } else {
                event_emitter.process_chunk(&chat_chunk, sink)?;
            }

            Ok(SseRecordOutcome::Continue)
        }
        SseDataRecord::RawJson(json_str) => {
            debug!("Non-chunk SSE event, passing through: {}", json_str);
            sink.send_raw_json(&json_str)?;

            if is_upstream_error_payload(&json_str) {
                Ok(SseRecordOutcome::UpstreamError)
            } else {
                Ok(SseRecordOutcome::Continue)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct FunctionCallEventState {
    output_index: usize,
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    added: bool,
    completed: bool,
}

fn process_non_mcp_function_call_chunk(
    chunk: &ChatCompletionStreamResponse,
    emitter: &mut ResponseStreamEventEmitter,
    sink: &impl ResponseEventSink,
    function_call_events: &mut BTreeMap<usize, FunctionCallEventState>,
) -> Result<(), String> {
    let Some(choice) = chunk.choices.first() else {
        return Ok(());
    };

    if let Some(tool_call_deltas) = &choice.delta.tool_calls {
        for delta in tool_call_deltas {
            let state = function_call_events
                .entry(delta.index as usize)
                .or_insert_with(|| {
                    let (output_index, item_id) =
                        emitter.allocate_output_index(OutputItemKind::FunctionCall);

                    FunctionCallEventState {
                        output_index,
                        item_id,
                        call_id: String::new(),
                        name: String::new(),
                        arguments: String::new(),
                        added: false,
                        completed: false,
                    }
                });

            if state.call_id.is_empty() {
                if let Some(delta_id) = &delta.id {
                    state.call_id = delta_id.clone();
                }
            }

            if let Some(function) = &delta.function {
                if let Some(delta_name) = &function.name {
                    state.name.push_str(delta_name);
                }

                if !state.added {
                    let item = json!({
                        "id": state.item_id,
                        "type": "function_call",
                        "call_id": state.call_id,
                        "name": state.name,
                        "status": "in_progress",
                        "arguments": ""
                    });
                    let event = emitter.emit_output_item_added(state.output_index, &item);
                    emitter.send_event(&event, sink)?;
                    state.added = true;
                }

                if let Some(delta_args) = &function.arguments {
                    if !delta_args.is_empty() {
                        state.arguments.push_str(delta_args);
                        let event = emitter.emit_function_call_arguments_delta(
                            state.output_index,
                            &state.item_id,
                            delta_args,
                        );
                        emitter.send_event(&event, sink)?;
                    }
                }
            }
        }
    }

    if choice.finish_reason.as_deref() == Some("tool_calls") {
        for state in function_call_events.values_mut() {
            if state.completed {
                continue;
            }

            if !state.added {
                let item = json!({
                    "id": state.item_id,
                    "type": "function_call",
                    "call_id": state.call_id,
                    "name": state.name,
                    "status": "in_progress",
                    "arguments": ""
                });
                let event = emitter.emit_output_item_added(state.output_index, &item);
                emitter.send_event(&event, sink)?;
                state.added = true;
            }

            let event = emitter.emit_function_call_arguments_done(
                state.output_index,
                &state.item_id,
                &state.arguments,
            );
            emitter.send_event(&event, sink)?;

            let item = json!({
                "id": state.item_id,
                "type": "function_call",
                "call_id": state.call_id,
                "name": state.name,
                "status": "completed",
                "arguments": state.arguments
            });
            let event = emitter.emit_output_item_done(state.output_index, &item);
            emitter.send_event(&event, sink)?;
            emitter.complete_output_item(state.output_index);
            state.completed = true;
        }
    }

    Ok(())
}

/// Response accumulator for streaming responses (non-MCP path)
struct StreamingResponseAccumulator {
    response_id: String,
    model: String,
    created_at: i64,
    content_buffer: String,
    reasoning_buffer: String,
    tool_calls: Vec<ResponseOutputItem>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
    original_request: ResponsesRequest,
}

impl StreamingResponseAccumulator {
    fn new(
        original_request: &ResponsesRequest,
        response_id: String,
        model: String,
        created_at: i64,
    ) -> Self {
        Self {
            response_id,
            model,
            created_at,
            content_buffer: String::new(),
            reasoning_buffer: String::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
            original_request: original_request.clone(),
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        if self.response_id.is_empty() {
            self.response_id.clone_from(&chunk.id);
            self.model.clone_from(&chunk.model);
            self.created_at = chunk.created as i64;
        }

        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                if self.tool_calls.is_empty() {
                    self.content_buffer.push_str(content);
                }
            }

            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_buffer.push_str(reasoning);
            }

            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    let index = delta.index as usize;

                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(ResponseOutputItem::FunctionToolCall {
                            id: generate_item_id("fc"),
                            call_id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            output: None,
                            status: "in_progress".to_string(),
                        });
                    }

                    if let ResponseOutputItem::FunctionToolCall {
                        call_id,
                        name,
                        arguments,
                        ..
                    } = &mut self.tool_calls[index]
                    {
                        if let Some(delta_id) = &delta.id {
                            if call_id.is_empty() {
                                call_id.clone_from(delta_id);
                            }
                        }
                        if let Some(function) = &delta.function {
                            if let Some(delta_name) = &function.name {
                                name.push_str(delta_name);
                            }
                            if let Some(delta_args) = &function.arguments {
                                arguments.push_str(delta_args);
                            }
                        }
                    }
                }
            }

            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn response_status(&self) -> ResponseStatus {
        match self.finish_reason.as_deref() {
            Some("stop") | Some("length") => ResponseStatus::Completed,
            Some("tool_calls") => ResponseStatus::InProgress,
            Some("failed") | Some("error") => ResponseStatus::Failed,
            _ => ResponseStatus::Completed,
        }
    }

    fn finalize(self) -> ResponsesResponse {
        let status = self.response_status();
        self.finalize_with_status(status)
    }

    fn finalize_with_status(self, status: ResponseStatus) -> ResponsesResponse {
        let mut output: Vec<ResponseOutputItem> = Vec::new();

        if !self.content_buffer.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", self.response_id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: self.content_buffer,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
                phase: None,
            });
        }

        if !self.reasoning_buffer.is_empty() {
            output.push(ResponseOutputItem::new_reasoning(
                format!("reasoning_{}", self.response_id),
                vec![],
                vec![ResponseReasoningContent::ReasoningText {
                    text: self.reasoning_buffer,
                }],
                Some("completed".to_string()),
            ));
        }

        output.extend(self.tool_calls);

        let usage = self.usage.as_ref().map(|u| {
            let usage_info = UsageInfo {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                prompt_tokens_details: None,
            };
            ResponsesUsage::Classic(usage_info)
        });

        ResponsesResponse::builder(&self.response_id, &self.model)
            .copy_from_request(&self.original_request)
            .created_at(self.created_at)
            .status(status)
            .output(output)
            .maybe_usage(usage)
            .build()
    }
}

fn is_upstream_error_payload(payload: &str) -> bool {
    serde_json::from_str::<Value>(payload)
        .ok()
        .and_then(|value| value.get("error").cloned())
        .is_some()
}

// ============================================================================
// MCP Streaming Path
// ============================================================================

/// Execute MCP tool loop with streaming support
///
/// This streams each iteration's response to the client while accumulating
/// to check for tool calls. If tool calls are found, executes them and
/// continues with the next streaming iteration.
pub(super) fn execute_tool_loop_streaming(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
) -> Response {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    let sink = SseResponseEventSink::new(tx.clone());

    let ctx_clone = ctx.clone();
    let original_request_clone = original_request.clone();

    #[expect(
        clippy::disallowed_methods,
        reason = "streaming task is fire-and-forget; client disconnect terminates it"
    )]
    tokio::spawn(async move {
        let result = execute_tool_loop_streaming_internal(
            &ctx_clone,
            current_request,
            &original_request_clone,
            params,
            mcp_servers,
            &sink,
        )
        .await;

        if let Err(e) = result {
            warn!("Streaming tool loop error: {}", e);
            let error_event = json!({
                "error": {
                    "message": e,
                    "type": "tool_loop_error"
                }
            });
            let _ = sink.send_raw_json(&error_event.to_string());
        }

        let _ = sink.send_done();
    });

    build_sse_response(rx)
}

pub(super) async fn execute_tool_loop_streaming_with_sink(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
    outbound_tx: mpsc::Sender<Message>,
) -> Result<ResponsesResponse, String> {
    let sink = WsResponseEventSink::new(outbound_tx);
    execute_tool_loop_streaming_internal(
        ctx,
        current_request,
        original_request,
        params,
        mcp_servers,
        &sink,
    )
    .await
}

/// Internal streaming tool loop implementation
async fn execute_tool_loop_streaming_internal(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
    sink: &impl ResponseEventSink,
) -> Result<ResponsesResponse, String> {
    let mut state = ToolLoopState::new(original_request.input.clone());
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    // Generate response ID first so we can use it for both emitter and session
    let response_id = format!("resp_{}", Uuid::now_v7());

    // Create session once — bundles orchestrator, request_ctx, server_keys, mcp_tools
    let session = McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &response_id);

    // Create response event emitter
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut emitter =
        ResponseStreamEventEmitter::new(response_id, current_request.model.clone(), created_at);
    emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events
    let event = emitter.emit_created();
    emitter.send_event(&event, sink)?;
    let event = emitter.emit_in_progress();
    emitter.send_event(&event, sink)?;

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&session);
    trace!(
        "Streaming: Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    // Flag to track if mcp_list_tools has been emitted
    let mut mcp_list_tools_emitted = false;

    loop {
        state.iteration += 1;

        // Record tool loop iteration metric
        Metrics::record_mcp_tool_iteration(&current_request.model);

        if state.iteration > DEFAULT_MAX_ITERATIONS {
            return Err(format!(
                "Tool loop exceeded maximum iterations ({DEFAULT_MAX_ITERATIONS})"
            ));
        }

        trace!("Streaming MCP tool loop iteration {}", state.iteration);

        // Emit mcp_list_tools as first output item (only once, on first iteration)
        if !mcp_list_tools_emitted {
            for binding in session.mcp_servers() {
                let tools_for_server = session.list_tools_for_server(&binding.server_key);

                emitter.emit_mcp_list_tools_sequence(&binding.label, &tools_for_server, sink)?;
            }
            mcp_list_tools_emitted = true;
        }

        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {e}"))?;

        // Prepare tools and tool_choice for this iteration (same logic as non-streaming)
        prepare_chat_tools_and_choice(&mut chat_request, &mcp_chat_tools, state.iteration);

        // Execute chat streaming
        let response = ctx
            .pipeline
            .execute_chat(
                Arc::new(chat_request),
                params.headers.clone(),
                params.model_id.clone(),
                ctx.components.clone(),
                Some(params.tenant_request_meta.clone()),
            )
            .await;

        // Convert chat stream to Responses API events while accumulating for tool call detection
        // Stream text naturally - it only appears on final iteration (tool iterations have empty content)
        let accumulated_response =
            convert_and_accumulate_stream(response.into_body(), &mut emitter, sink).await?;

        // Check for tool calls (extract all of them for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&accumulated_response);

        if !tool_calls.is_empty() {
            trace!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Separate MCP and function tool calls using session-exposed names.
            let (mcp_tool_calls, function_tool_calls): (Vec<ExtractedToolCall>, Vec<_>) =
                tool_calls
                    .into_iter()
                    .partition(|tc| session.has_exposed_tool(tc.name.as_str()));

            trace!(
                "Separated tool calls: {} MCP, {} function",
                mcp_tool_calls.len(),
                function_tool_calls.len()
            );

            // Check combined limit (only count MCP tools since function tools will be returned)
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    DEFAULT_MAX_ITERATIONS
                );
                let mut final_response = emitter.finalize_with_status(
                    accumulated_response.usage.clone(),
                    ResponseStatus::Completed,
                );
                final_response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                let usage_json = accumulated_response.usage.as_ref().map(|u| {
                    json!({
                        "input_tokens": u.prompt_tokens,
                        "output_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens
                    })
                });
                let mut event = emitter.emit_completed(usage_json.as_ref());
                if let Some(response) = event.get_mut("response") {
                    response["incomplete_details"] = json!({ "reason": "max_tool_calls" });
                }
                emitter.send_event(&event, sink)?;

                persist_response_if_needed(
                    ctx.conversation_storage.clone(),
                    ctx.conversation_item_storage.clone(),
                    ctx.response_storage.clone(),
                    &final_response,
                    original_request,
                    ctx.request_context.clone(),
                )
                .await;

                return Ok(final_response);
            }

            // Process each MCP tool call
            for tool_call in mcp_tool_calls {
                state.total_calls += 1;

                trace!(
                    "Executing tool call {}/{}: {} (call_id: {})",
                    state.total_calls,
                    state.total_calls,
                    tool_call.name,
                    tool_call.call_id
                );

                let response_format = openai_bridge::lookup_tool_format(
                    &session,
                    &ctx.mcp_format_registry,
                    &tool_call.name,
                );

                // Use emitter helpers to determine correct type and allocate index
                let item_type =
                    ResponseStreamEventEmitter::type_str_for_format(Some(&response_format));
                let resolved_label = session.resolve_tool_server_label(&tool_call.name);

                // Allocate output_index with the format's id-prefix discriminator
                // (e.g. `ws_…` for web_search_call); see FormatDescriptor.
                let (output_index, item_id) =
                    emitter.allocate_output_index_for_format(Some(response_format));

                // Build initial tool call item
                let mut item = json!({
                    "id": item_id,
                    "type": item_type,
                    "name": tool_call.name,
                    "status": "in_progress",
                    "arguments": ""
                });
                attach_mcp_server_label(
                    &mut item,
                    Some(resolved_label.as_str()),
                    Some(&response_format),
                );

                // Emit output_item.added
                let event = emitter.emit_output_item_added(output_index, &item);
                emitter.send_event(&event, sink)?;

                // Emit tool_call.in_progress
                let event =
                    emitter.emit_tool_call_in_progress(output_index, &item_id, response_format);
                emitter.send_event(&event, sink)?;

                // Emit arguments events for mcp_call only (skip for builtin tools)
                if matches!(response_format, ResponseFormat::Passthrough) {
                    // Emit mcp_call_arguments.delta (simulate streaming by sending full arguments)
                    let event = emitter.emit_mcp_call_arguments_delta(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;

                    // Emit mcp_call_arguments.done
                    let event = emitter.emit_mcp_call_arguments_done(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;
                }

                // Emit searching/interpreting event for builtin tools
                if let Some(event) =
                    emitter.emit_tool_call_searching(output_index, &item_id, response_format)
                {
                    emitter.send_event(&event, sink)?;
                }

                // Execute the MCP tool
                trace!(
                    "Calling MCP tool '{}' with args: {}",
                    tool_call.name,
                    tool_call.arguments
                );
                // Parse arguments to Value, coercing scalar/array/null payloads
                // to an empty object so hosted-tool override merge can actually
                // apply. `apply_hosted_tool_overrides` is a no-op on non-objects;
                // silently dropping caller-declared config would be surprising.
                let mut arguments = match serde_json::from_str::<Value>(&tool_call.arguments) {
                    Ok(Value::Object(map)) => Value::Object(map),
                    _ => json!({}),
                };
                prepare_hosted_dispatch_args(
                    &mut arguments,
                    response_format,
                    original_request.tools.as_deref().unwrap_or(&[]),
                    original_request.user.as_deref(),
                );

                // Execute the single tool via the normalized MCP execution API.
                // This avoids custom serialization and manual re-transformation in streaming paths.
                let tool_output = session
                    .execute_tool(ToolExecutionInput {
                        call_id: tool_call.call_id.clone(),
                        tool_name: tool_call.name.clone(),
                        arguments,
                    })
                    .await;

                let success = !tool_output.is_error;
                let output_str = tool_output.output.to_string();

                let output_item =
                    openai_bridge::transform_tool_output(&tool_output, response_format);
                let mut item_done = serde_json::to_value(&output_item).unwrap_or_else(|e| {
                    warn!(
                        tool = %tool_output.tool_name,
                        error = %e,
                        "Failed to serialize transformed output item; falling back to a minimal stub",
                    );
                    json!({
                        "id": item_id,
                        "type": item_type,
                        "status": if success { "completed" } else { "failed" },
                    })
                });
                // Override the typed item's id so output_item.done matches the
                // streaming-allocated id used by the earlier output_item.added.
                if let Some(obj) = item_done.as_object_mut() {
                    obj.insert("id".to_string(), json!(&item_id));
                }
                attach_mcp_server_label(
                    &mut item_done,
                    Some(tool_output.server_label.as_str()),
                    Some(&response_format),
                );

                if success {
                    let event =
                        emitter.emit_tool_call_completed(output_index, &item_id, response_format);
                    emitter.send_event(&event, sink)?;
                } else {
                    let err_text = tool_output
                        .error_message
                        .clone()
                        .unwrap_or_else(|| output_str.clone());
                    warn!("Tool execution returned error: {}", err_text);

                    // `response.mcp_call.failed` is the only `*.failed` event
                    // in the Responses API. Hosted-builtin families close via
                    // `*.completed` to mirror OpenAI cloud's wire shape;
                    // the failure context (when present) lives in the item
                    // content.
                    if matches!(response_format, ResponseFormat::Passthrough) {
                        let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err_text);
                        emitter.send_event(&event, sink)?;
                    } else {
                        let event = emitter.emit_tool_call_completed(
                            output_index,
                            &item_id,
                            response_format,
                        );
                        emitter.send_event(&event, sink)?;
                    }
                }

                let event = emitter.emit_output_item_done(output_index, &item_done);
                emitter.send_event(&event, sink)?;
                emitter.complete_output_item(output_index);

                Metrics::record_mcp_tool_duration(
                    &current_request.model,
                    &tool_output.tool_name,
                    tool_output.duration,
                );
                Metrics::record_mcp_tool_call(
                    &current_request.model,
                    &tool_output.tool_name,
                    if success {
                        metrics_labels::RESULT_SUCCESS
                    } else {
                        metrics_labels::RESULT_ERROR
                    },
                );

                state.record_call(
                    tool_output.call_id,
                    tool_output.tool_name,
                    tool_output.arguments_str,
                    output_str,
                    output_item,
                    success,
                );
            }

            // If there are function tool calls, emit events and exit MCP loop
            if !function_tool_calls.is_empty() {
                trace!(
                    "Found {} function tool call(s) - emitting events and exiting MCP loop",
                    function_tool_calls.len()
                );

                // Emit function_tool_call events for each function tool
                for tool_call in function_tool_calls {
                    // Allocate output_index for this function_tool_call item
                    let (output_index, item_id) =
                        emitter.allocate_output_index(OutputItemKind::FunctionCall);

                    // Build initial function_call item
                    let item = json!({
                        "id": item_id,
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "in_progress",
                        "arguments": ""
                    });

                    // Emit output_item.added
                    let event = emitter.emit_output_item_added(output_index, &item);
                    emitter.send_event(&event, sink)?;

                    // Emit function_call_arguments.delta
                    let event = emitter.emit_function_call_arguments_delta(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;

                    // Emit function_call_arguments.done
                    let event = emitter.emit_function_call_arguments_done(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, sink)?;

                    // Build complete item
                    let item_complete = json!({
                        "id": item_id,
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "completed",
                        "arguments": tool_call.arguments
                    });

                    // Emit output_item.done
                    let event = emitter.emit_output_item_done(output_index, &item_complete);
                    emitter.send_event(&event, sink)?;

                    emitter.complete_output_item(output_index);
                }

                let usage_json = accumulated_response.usage.as_ref().map(|u| {
                    json!({
                        "input_tokens": u.prompt_tokens,
                        "output_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens
                    })
                });
                let event = emitter
                    .emit_completed_with_status(ResponseStatus::InProgress, usage_json.as_ref());
                emitter.send_event(&event, sink)?;

                let final_response = emitter.finalize_with_status(
                    accumulated_response.usage.clone(),
                    ResponseStatus::InProgress,
                );
                persist_response_if_needed(
                    ctx.conversation_storage.clone(),
                    ctx.conversation_item_storage.clone(),
                    ctx.response_storage.clone(),
                    &final_response,
                    original_request,
                    ctx.request_context.clone(),
                )
                .await;

                return Ok(final_response);
            }

            // Build next request with conversation history
            current_request = build_next_request(&state, current_request);

            continue;
        }

        // No tool calls, this is the final response
        trace!("No tool calls found, ending streaming MCP loop");

        // Check for reasoning content
        let reasoning_content = accumulated_response
            .choices
            .first()
            .and_then(|c| c.message.reasoning_content.clone());

        // Emit reasoning item if present
        if let Some(reasoning) = reasoning_content {
            if !reasoning.is_empty() {
                emitter.emit_reasoning_item(sink, Some(reasoning))?;
            }
        }

        // Text message events already emitted naturally by process_chunk during stream processing
        // (OpenAI router approach - text only appears on final iteration when no tool calls)

        // Emit final response.completed event
        let usage_json = accumulated_response.usage.as_ref().map(|u| {
            json!({
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens
            })
        });
        let event = emitter.emit_completed(usage_json.as_ref());
        emitter.send_event(&event, sink)?;

        let final_response = emitter.finalize(accumulated_response.usage.clone());
        persist_response_if_needed(
            ctx.conversation_storage.clone(),
            ctx.conversation_item_storage.clone(),
            ctx.response_storage.clone(),
            &final_response,
            original_request,
            ctx.request_context.clone(),
        )
        .await;

        return Ok(final_response);
    }
}

/// Convert chat stream to Responses API events while accumulating for tool call detection
async fn convert_and_accumulate_stream(
    body: Body,
    emitter: &mut ResponseStreamEventEmitter,
    sink: &impl ResponseEventSink,
) -> Result<ChatCompletionResponse, String> {
    let mut accumulator = ChatResponseAccumulator::new();
    let mut stream = body.into_data_stream();
    let mut parser = SseBodyParser::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {e}"))?;
        parser.push(&chunk);

        while let Some(record) = parser.next_record() {
            match record {
                SseDataRecord::Done => return Ok(accumulator.finalize()),
                SseDataRecord::ChatChunk(chat_chunk) => {
                    emitter.process_chunk(&chat_chunk, sink)?;
                    accumulator.process_chunk(&chat_chunk);
                }
                SseDataRecord::RawJson(_) => {
                    // MCP path ignores non-chat payloads; terminal router errors
                    // are surfaced before this function is entered.
                }
            }
        }
    }

    if let Some(SseDataRecord::ChatChunk(chat_chunk)) = parser.flush() {
        emitter.process_chunk(&chat_chunk, sink)?;
        accumulator.process_chunk(&chat_chunk);
    }

    Ok(accumulator.finalize())
}

/// Accumulates chat streaming chunks into complete ChatCompletionResponse
struct ChatResponseAccumulator {
    id: String,
    model: String,
    content: String,
    reasoning_content: Option<String>,
    tool_calls: HashMap<usize, ToolCall>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

impl ChatResponseAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: HashMap::new(),
            finish_reason: None,
            usage: None,
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        if !chunk.id.is_empty() {
            self.id.clone_from(&chunk.id);
        }
        if !chunk.model.is_empty() {
            self.model.clone_from(&chunk.model);
        }

        if let Some(choice) = chunk.choices.first() {
            // Accumulate content
            if let Some(content) = &choice.delta.content {
                self.content.push_str(content);
            }

            // Accumulate reasoning content
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_content
                    .get_or_insert_with(String::new)
                    .push_str(reasoning);
            }

            // Accumulate tool calls
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    let index = delta.index as usize;
                    let entry = self.tool_calls.entry(index).or_insert_with(|| ToolCall {
                        id: String::new(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: String::new(),
                            arguments: Some(String::new()),
                        },
                    });

                    if let Some(id) = &delta.id {
                        entry.id.clone_from(id);
                    }
                    if let Some(function) = &delta.function {
                        if let Some(name) = &function.name {
                            entry.function.name.clone_from(name);
                        }
                        if let Some(args) = &function.arguments {
                            if let Some(ref mut existing_args) = entry.function.arguments {
                                existing_args.push_str(args);
                            }
                        }
                    }
                }
            }

            // Capture finish reason
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

        // Update usage
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ChatCompletionResponse {
        let mut tool_calls_vec: Vec<_> = self.tool_calls.into_iter().collect();
        tool_calls_vec.sort_by_key(|(index, _)| *index);
        let tool_calls: Vec<_> = tool_calls_vec.into_iter().map(|(_, call)| call).collect();

        ChatCompletionResponse::builder(&self.id, &self.model)
            .choices(vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: if self.content.is_empty() {
                        None
                    } else {
                        Some(self.content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    reasoning_content: self.reasoning_content,
                },
                finish_reason: self.finish_reason,
                logprobs: None,
                matched_stop: None,
                hidden_states: None,
            }])
            .maybe_usage(self.usage)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::body::Body;
    use bytes::Bytes;
    use futures_util::stream;
    use serde_json::Value;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage, ResponseId,
    };

    use super::*;

    #[derive(Clone, Default)]
    struct RecordingSink {
        events: Arc<Mutex<Vec<Value>>>,
    }

    impl RecordingSink {
        fn events(&self) -> Vec<Value> {
            self.events.lock().unwrap().clone()
        }
    }

    impl ResponseEventSink for RecordingSink {
        fn send_event(&self, event: &Value) -> Result<(), String> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }

        fn send_raw_json(&self, payload: &str) -> Result<(), String> {
            let value = serde_json::from_str::<Value>(payload)
                .map_err(|err| format!("failed to parse test payload: {err}"))?;
            self.events.lock().unwrap().push(value);
            Ok(())
        }
    }

    fn test_responses_request() -> ResponsesRequest {
        ResponsesRequest {
            background: Some(false),
            input: openai_protocol::responses::ResponseInput::Text(
                "trigger upstream error".to_string(),
            ),
            max_output_tokens: Some(64),
            model: "mock-model".to_string(),
            parallel_tool_calls: Some(true),
            store: Some(true),
            stream: Some(true),
            temperature: Some(0.0),
            top_logprobs: Some(0),
            truncation: Some(openai_protocol::responses::Truncation::Disabled),
            request_id: Some("resp_stream_upstream_error".to_string()),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            ..ResponsesRequest::default()
        }
    }

    #[tokio::test]
    async fn test_process_and_transform_stream_does_not_emit_completed_after_upstream_error() {
        let body = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(
                "data: {\"error\":{\"message\":\"upstream exploded\",\"type\":\"internal_error\"}}\n\n",
            )),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ]));
        let response_storage = Arc::new(MemoryResponseStorage::new());
        let conversation_storage = Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());
        let sink = RecordingSink::default();

        let final_response = process_and_transform_stream(
            body,
            test_responses_request(),
            response_storage.clone(),
            conversation_storage,
            conversation_item_storage,
            None,
            &sink,
        )
        .await
        .expect("stream processing should succeed after forwarding upstream error");

        let events = sink.events();
        let event_types: Vec<_> = events
            .iter()
            .filter_map(|event| event.get("type").and_then(|value| value.as_str()))
            .collect();

        assert_eq!(
            event_types,
            vec!["response.created", "response.in_progress"],
            "only the synthetic start events should be emitted before the upstream error",
        );
        assert!(
            events.iter().any(|event| {
                event
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(|value| value.as_str())
                    == Some("upstream exploded")
            }),
            "the upstream error payload should be forwarded as-is",
        );
        assert_eq!(final_response.status, ResponseStatus::Failed);

        let response_id = ResponseId::from(final_response.id.as_str());
        assert!(
            response_storage
                .get_response(&response_id)
                .await
                .expect("storage lookup should succeed")
                .is_none(),
            "failed upstream streams should not be persisted",
        );
    }

    #[tokio::test]
    async fn test_process_and_transform_stream_emits_function_call_events_for_non_mcp_streams() {
        let body = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_test\",\"function\":{\"name\":\"get_weather\"},\"type\":\"function\"}]},\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"location\\\":\\\"Berlin\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n\n",
            )),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ]));
        let sink = RecordingSink::default();

        let final_response = process_and_transform_stream(
            body,
            test_responses_request(),
            Arc::new(MemoryResponseStorage::new()),
            Arc::new(MemoryConversationStorage::new()),
            Arc::new(MemoryConversationItemStorage::new()),
            None,
            &sink,
        )
        .await
        .expect("tool-call stream processing should succeed");

        let events = sink.events();
        let event_types: Vec<_> = events
            .iter()
            .filter_map(|event| event.get("type").and_then(|value| value.as_str()))
            .collect();

        assert_eq!(
            event_types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        );
        assert_eq!(events[2]["item"]["type"], "function_call");
        assert_eq!(events[2]["item"]["call_id"], "call_test");
        assert_eq!(events[3]["delta"], "{\"location\":\"Berlin\"}");
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["type"],
            "function_call"
        );
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["call_id"],
            "call_test"
        );
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["arguments"],
            "{\"location\":\"Berlin\"}"
        );
        assert_eq!(events.last().unwrap()["response"]["status"], "in_progress");

        assert_eq!(final_response.status, ResponseStatus::InProgress);
        assert_eq!(final_response.output.len(), 1);
        let serialized_output = serde_json::to_value(&final_response.output[0]).unwrap();
        assert_eq!(serialized_output["type"], "function_call");
        assert_eq!(serialized_output["call_id"], "call_test");
        assert_eq!(serialized_output["arguments"], "{\"location\":\"Berlin\"}");
    }

    #[tokio::test]
    async fn test_process_and_transform_stream_keeps_function_call_id_stable_across_repeated_deltas(
    ) {
        let body = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_test\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\"},\"type\":\"function\"}]},\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_test\",\"function\":{\"arguments\":\"\\\"location\\\":\\\"Berlin\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n\n",
            )),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ]));
        let sink = RecordingSink::default();

        let final_response = process_and_transform_stream(
            body,
            test_responses_request(),
            Arc::new(MemoryResponseStorage::new()),
            Arc::new(MemoryConversationStorage::new()),
            Arc::new(MemoryConversationItemStorage::new()),
            None,
            &sink,
        )
        .await
        .expect("tool-call stream processing should succeed");

        let events = sink.events();
        assert_eq!(events[2]["item"]["call_id"], "call_test");
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["call_id"],
            "call_test"
        );

        let serialized_output = serde_json::to_value(&final_response.output[0]).unwrap();
        assert_eq!(serialized_output["call_id"], "call_test");
        assert_eq!(serialized_output["arguments"], "{\"location\":\"Berlin\"}");
    }

    #[tokio::test]
    async fn test_process_and_transform_stream_emits_added_for_zero_argument_function_calls() {
        let body = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_zero\",\"function\":{\"name\":\"ping\"},\"type\":\"function\"}]},\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"id\":\"chatcmpl_test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":1,\"total_tokens\":9}}\n\n",
            )),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ]));
        let sink = RecordingSink::default();

        let final_response = process_and_transform_stream(
            body,
            test_responses_request(),
            Arc::new(MemoryResponseStorage::new()),
            Arc::new(MemoryConversationStorage::new()),
            Arc::new(MemoryConversationItemStorage::new()),
            None,
            &sink,
        )
        .await
        .expect("zero-arg tool-call stream processing should succeed");

        let events = sink.events();
        let event_types: Vec<_> = events
            .iter()
            .filter_map(|event| event.get("type").and_then(|value| value.as_str()))
            .collect();

        assert_eq!(
            event_types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        );
        assert_eq!(events[2]["item"]["call_id"], "call_zero");
        assert_eq!(events[2]["item"]["name"], "ping");
        assert_eq!(events[3]["arguments"], "");
        assert_eq!(events[4]["item"]["arguments"], "");

        let serialized_output = serde_json::to_value(&final_response.output[0]).unwrap();
        assert_eq!(serialized_output["call_id"], "call_zero");
        assert_eq!(serialized_output["name"], "ping");
        assert_eq!(serialized_output["arguments"], "");
    }

    #[tokio::test]
    async fn test_convert_and_accumulate_stream_handles_coalesced_sse_records() {
        let body = Body::from_stream(stream::iter(vec![Ok::<_, std::io::Error>(
            Bytes::from(
                concat!(
                    "data: {\"id\":\"chatcmpl_mcp\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n",
                    "data: {\"id\":\"chatcmpl_mcp\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
                    "data: {\"id\":\"chatcmpl_mcp\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
                    "data: [DONE]\n\n"
                ),
            ),
        )]));

        let sink = RecordingSink::default();
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_mcp_test".into(), "mock-model".into(), 1);
        let created = emitter.emit_created();
        emitter.send_event(&created, &sink).unwrap();
        let in_prog = emitter.emit_in_progress();
        emitter.send_event(&in_prog, &sink).unwrap();

        let result = convert_and_accumulate_stream(body, &mut emitter, &sink)
            .await
            .expect("coalesced MCP stream should succeed");

        assert_eq!(result.choices[0].message.content.as_deref(), Some("hello"));
        assert!(result.usage.is_some());

        let events = sink.events();
        let text_deltas: Vec<&str> = events
            .iter()
            .filter(|e| {
                e.get("type").and_then(|t| t.as_str()) == Some("response.output_text.delta")
            })
            .filter_map(|e| e.get("delta").and_then(|d| d.as_str()))
            .collect();
        assert_eq!(text_deltas, vec!["hel", "lo"]);
    }

    #[tokio::test]
    async fn test_convert_and_accumulate_stream_handles_split_sse_records() {
        let first_half = "data: {\"id\":\"chatcmpl_split\",\"object\":\"chat.completion.chunk\",";
        let second_half = "\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"split\"},\"finish_reason\":null}]}\n\n\
                           data: {\"id\":\"chatcmpl_split\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                           data: [DONE]\n\n";

        let body = Body::from_stream(stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from(first_half)),
            Ok(Bytes::from(second_half)),
        ]));

        let sink = RecordingSink::default();
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_split_test".into(), "mock-model".into(), 1);
        let created = emitter.emit_created();
        emitter.send_event(&created, &sink).unwrap();
        let in_prog = emitter.emit_in_progress();
        emitter.send_event(&in_prog, &sink).unwrap();

        let result = convert_and_accumulate_stream(body, &mut emitter, &sink)
            .await
            .expect("split MCP stream should succeed");

        assert_eq!(result.choices[0].message.content.as_deref(), Some("split"));

        let events = sink.events();
        let text_deltas: Vec<&str> = events
            .iter()
            .filter(|e| {
                e.get("type").and_then(|t| t.as_str()) == Some("response.output_text.delta")
            })
            .filter_map(|e| e.get("delta").and_then(|d| d.as_str()))
            .collect();
        assert_eq!(text_deltas, vec!["split"]);
    }
}

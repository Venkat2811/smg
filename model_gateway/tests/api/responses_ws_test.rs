use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        FromRequestParts,
    },
    http::{HeaderMap, Request},
    response::{IntoResponse, Response},
};
use futures_util::{SinkExt, StreamExt};
use openai_protocol::responses::{
    ResponseContentPart, ResponseInputOutputItem, ResponseOutputItem, ResponseStatus,
    ResponsesRequest, ResponsesResponse,
};
use smg::{
    middleware::TenantRequestMeta,
    routers::{
        ws_responses::{
            serve_responses_ws_with_config, CachedWsResponse, WsClientError,
            WsResponseCreateOptions, WsResponsesExecutor, WsRuntimeConfig,
        },
        RouterTrait,
    },
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, Mutex, Notify},
};
use tokio_tungstenite::connect_async;

use crate::common::test_app::{create_test_app_context, create_test_app_with_context};

#[derive(Clone)]
struct StubWsExecutor {
    gate: Option<Arc<Notify>>,
}

impl StubWsExecutor {
    fn immediate() -> Self {
        Self { gate: None }
    }

    fn gated(gate: Arc<Notify>) -> Self {
        Self { gate: Some(gate) }
    }
}

#[async_trait]
impl WsResponsesExecutor for StubWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        _options: WsResponseCreateOptions,
        _cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": "resp_ws_test",
                "object": "response",
                "status": "in_progress",
                "model": request.model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        if let Some(gate) = &self.gate {
            gate.notified().await;
        }

        let response = ResponsesResponse::builder("resp_ws_test", request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: "msg_ws_test".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: "stub websocket output".to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
                phase: None,
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response.clone(),
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        Ok(CachedWsResponse {
            response,
            input_items: vec![ResponseInputOutputItem::Message {
                id: "msg_user_ws_test".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello websocket".to_string(),
                }],
                status: Some("completed".to_string()),
                phase: None,
            }],
        })
    }
}

#[derive(Clone, Default)]
struct SemanticWsExecutor {
    durable_store: Arc<Mutex<HashMap<String, CachedWsResponse>>>,
}

#[async_trait]
impl WsResponsesExecutor for SemanticWsExecutor {
    async fn execute_response_create(
        &self,
        _headers: HeaderMap,
        request: ResponsesRequest,
        options: WsResponseCreateOptions,
        cached_response: Option<CachedWsResponse>,
        outbound_tx: mpsc::Sender<Message>,
    ) -> Result<CachedWsResponse, WsClientError> {
        if request.conversation.is_some() {
            return Err(WsClientError::new(
                "unsupported_parameter",
                "The `conversation` field is not supported in WebSocket Responses V1.",
            ));
        }

        if options.generate == Some(false) {
            let response_id = format!("resp_ws_{}", uuid::Uuid::new_v4().simple());
            let response = ResponsesResponse::builder(response_id.clone(), request.model.clone())
                .copy_from_request(&request)
                .status(ResponseStatus::Completed)
                .output(vec![])
                .build();

            let created = serde_json::json!({
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": request.model,
                    "output": []
                }
            });
            let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));
            let completed = serde_json::json!({
                "type": "response.completed",
                "response": response.clone()
            });
            let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

            return Ok(CachedWsResponse {
                response,
                input_items: vec![],
            });
        }

        let continuation_source = if let Some(previous_id) = request.previous_response_id.as_deref()
        {
            if cached_response
                .as_ref()
                .is_some_and(|cached| cached.response.id == previous_id)
            {
                "cached"
            } else if let Some(stored) = self.durable_store.lock().await.get(previous_id).cloned() {
                let _ = stored;
                "durable"
            } else {
                return Err(
                    WsClientError::new(
                        "previous_response_not_found",
                        format!(
                            "Previous response '{}' was not found in the current session or durable storage.",
                            previous_id
                        ),
                    )
                    .with_param("previous_response_id"),
                );
            }
        } else {
            "fresh"
        };

        let response_id = format!("resp_ws_{}", uuid::Uuid::new_v4().simple());
        let response_text = format!("{continuation_source} websocket output");

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": request.model,
                "output": []
            }
        });
        let _ = outbound_tx.try_send(Message::Text(created.to_string().into()));

        let response = ResponsesResponse::builder(response_id.clone(), request.model.clone())
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .output(vec![ResponseOutputItem::Message {
                id: format!("msg_{response_id}"),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: response_text,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
                phase: None,
            }])
            .build();

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response.clone(),
        });
        let _ = outbound_tx.try_send(Message::Text(completed.to_string().into()));

        let cached = CachedWsResponse {
            response: response.clone(),
            input_items: vec![ResponseInputOutputItem::Message {
                id: format!("msg_user_{response_id}"),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello websocket".to_string(),
                }],
                status: Some("completed".to_string()),
                phase: None,
            }],
        };

        if request.store.unwrap_or(true) {
            self.durable_store
                .lock()
                .await
                .insert(response_id, cached.clone());
        }

        Ok(cached)
    }
}

#[derive(Clone)]
struct StubWsRouter {
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
}

impl fmt::Debug for StubWsRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StubWsRouter")
    }
}

#[async_trait]
impl RouterTrait for StubWsRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_responses_ws(
        &self,
        req: Request<Body>,
        _tenant_meta: TenantRequestMeta,
    ) -> Response {
        let (mut parts, _body) = req.into_parts();
        let headers = parts.headers.clone();
        let executor = self.executor.clone();
        let runtime_config = self.runtime_config.clone();

        match WebSocketUpgrade::from_request_parts(&mut parts, &()).await {
            Ok(ws) => ws
                .on_upgrade(move |socket: WebSocket| async move {
                    serve_responses_ws_with_config(socket, headers, executor, runtime_config).await;
                })
                .into_response(),
            Err(err) => err.into_response(),
        }
    }

    fn router_type(&self) -> &'static str {
        "stub-ws"
    }
}

async fn build_stub_app(executor: Arc<dyn WsResponsesExecutor>) -> axum::Router {
    build_stub_app_with_runtime_config(executor, WsRuntimeConfig::default()).await
}

async fn build_stub_app_with_runtime_config(
    executor: Arc<dyn WsResponsesExecutor>,
    runtime_config: WsRuntimeConfig,
) -> axum::Router {
    let ctx = create_test_app_context().await;
    let router = Arc::new(StubWsRouter {
        executor,
        runtime_config,
    });
    create_test_app_with_context(router, ctx)
}

async fn serve_app(app: axum::Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("ws://{addr}/v1/responses")
}

async fn recv_json(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> serde_json::Value {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(3), socket.next())
            .await
            .expect("timed out waiting for websocket message")
            .expect("websocket stream ended")
            .expect("websocket receive failed");

        match message {
            tokio_tungstenite::tungstenite::Message::Text(text) => {
                return serde_json::from_str(text.as_ref()).expect("message should be valid JSON");
            }
            tokio_tungstenite::tungstenite::Message::Ping(_) => continue,
            tokio_tungstenite::tungstenite::Message::Pong(_) => continue,
            tokio_tungstenite::tungstenite::Message::Close(frame) => {
                panic!("unexpected websocket close frame: {:?}", frame)
            }
            other => panic!("unexpected websocket message: {:?}", other),
        }
    }
}

async fn send_ws_request_and_collect(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    request: serde_json::Value,
) -> Vec<serde_json::Value> {
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .unwrap();

    let mut events = Vec::new();
    loop {
        let event = recv_json(socket).await;
        let is_terminal = matches!(
            event["type"].as_str(),
            Some("response.completed") | Some("error")
        );
        events.push(event);
        if is_terminal {
            break;
        }
    }

    events
}

fn ws_create_request(response_fields: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(mut request) = response_fields else {
        panic!("response.create request fields must be a JSON object");
    };
    request.insert(
        "type".to_string(),
        serde_json::Value::String("response.create".to_string()),
    );
    serde_json::Value::Object(request)
}

fn ws_error_code(event: &serde_json::Value) -> &str {
    event
        .pointer("/error/code")
        .and_then(|value| value.as_str())
        .or_else(|| event.get("code").and_then(|value| value.as_str()))
        .unwrap_or("")
}

fn ws_error_param(event: &serde_json::Value) -> Option<&str> {
    event
        .pointer("/error/param")
        .and_then(|value| value.as_str())
}

#[tokio::test]
async fn test_responses_ws_smoke() {
    let app = build_stub_app(Arc::new(StubWsExecutor::immediate())).await;
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(url).await.unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "hello"
        })),
    )
    .await;

    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(events[1]["type"], "response.completed");
}

#[tokio::test]
async fn test_responses_ws_rejects_invalid_json() {
    let app = build_stub_app(Arc::new(StubWsExecutor::immediate())).await;
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(url).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text("{".into()))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(event["type"], "error");
    assert_eq!(ws_error_code(&event), "invalid_json");
}

#[tokio::test]
async fn test_responses_ws_rejects_unsupported_event() {
    let app = build_stub_app(Arc::new(StubWsExecutor::immediate())).await;
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(url).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::json!({ "type": "response.cancel" })
                .to_string()
                .into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(ws_error_code(&event), "unsupported_event");
}

#[tokio::test]
async fn test_responses_ws_rejects_binary_messages() {
    let app = build_stub_app(Arc::new(StubWsExecutor::immediate())).await;
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(url).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Binary(
            vec![1, 2, 3].into(),
        ))
        .await
        .unwrap();

    let event = recv_json(&mut socket).await;
    assert_eq!(ws_error_code(&event), "unsupported_message_type");
}

#[tokio::test]
async fn test_responses_ws_same_socket_store_false_continuation_uses_cache() {
    let app = build_stub_app(Arc::new(SemanticWsExecutor::default())).await;
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(url).await.unwrap();

    let first_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "hello",
            "store": false
        })),
    )
    .await;
    let first_response_id = first_events[1]["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let second_events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "previous_response_id": first_response_id,
            "input": "continue",
            "store": false
        })),
    )
    .await;

    let output_text = second_events[1]["response"]["output"][0]["content"][0]["text"]
        .as_str()
        .unwrap();
    assert_eq!(output_text, "cached websocket output");
}

#[tokio::test]
async fn test_responses_ws_new_socket_store_false_continuation_fails() {
    let app = build_stub_app(Arc::new(SemanticWsExecutor::default())).await;
    let url = serve_app(app).await;

    let (mut first_socket, _) = connect_async(&url).await.unwrap();
    let first_events = send_ws_request_and_collect(
        &mut first_socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "hello",
            "store": false
        })),
    )
    .await;
    let first_response_id = first_events[1]["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    first_socket.close(None).await.unwrap();

    let (mut second_socket, _) = connect_async(url).await.unwrap();
    let second_events = send_ws_request_and_collect(
        &mut second_socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "previous_response_id": first_response_id,
            "input": "continue",
            "store": false
        })),
    )
    .await;

    assert_eq!(second_events[0]["type"], "error");
    assert_eq!(
        ws_error_code(&second_events[0]),
        "previous_response_not_found"
    );
    assert_eq!(
        ws_error_param(&second_events[0]),
        Some("previous_response_id")
    );
}

#[tokio::test]
async fn test_responses_ws_rejects_conversation_in_v1() {
    let app = build_stub_app(Arc::new(SemanticWsExecutor::default())).await;
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(url).await.unwrap();

    let events = send_ws_request_and_collect(
        &mut socket,
        ws_create_request(serde_json::json!({
            "model": "mock-model",
            "input": "hello",
            "conversation": "conv_123"
        })),
    )
    .await;

    assert_eq!(events[0]["type"], "error");
    assert_eq!(ws_error_code(&events[0]), "unsupported_parameter");
}

#[tokio::test]
async fn test_responses_ws_enforces_single_inflight_request() {
    let gate = Arc::new(Notify::new());
    let app = build_stub_app(Arc::new(StubWsExecutor::gated(gate.clone()))).await;
    let url = serve_app(app).await;
    let (mut socket, _) = connect_async(url).await.unwrap();

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            ws_create_request(serde_json::json!({
                "model": "mock-model",
                "input": "hello"
            }))
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let created = recv_json(&mut socket).await;
    assert_eq!(created["type"], "response.created");

    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            ws_create_request(serde_json::json!({
                "model": "mock-model",
                "input": "second"
            }))
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let error = recv_json(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(ws_error_code(&error), "concurrent_response_create");

    gate.notify_waiters();
    let completed = recv_json(&mut socket).await;
    assert_eq!(completed["type"], "response.completed");
}

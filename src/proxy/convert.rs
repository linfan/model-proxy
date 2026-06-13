use serde_json::Value;

use crate::types::anthropic::*;
use crate::types::openai::*;

use super::ThinkingSettings;

// ── OpenAI → Anthropic request ──

pub fn openai_to_anthropic_request(
    req: &ChatCompletionRequest,
    remote_model: &str,
    thinking: &ThinkingSettings,
) -> MessagesRequest {
    let mut system: Option<SystemContent> = None;
    let mut messages: Vec<AnthropicMessage> = Vec::new();

    for msg in &req.messages {
        if msg.role == "system" {
            let text = content_to_string(&msg.content);
            system = Some(SystemContent::Text(text));
            continue;
        }

        let role = match msg.role.as_str() {
            "assistant" => "assistant",
            "tool" => "user",
            _ => "user",
        };

        let content = if msg.role == "tool" {
            let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
            let text = content_to_string(&msg.content);
            MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id,
                content: Value::String(text),
                is_error: None,
            }])
        } else if msg.role == "assistant" {
            if let Some(tool_calls) = &msg.tool_calls {
                let mut blocks: Vec<ContentBlock> = Vec::new();
                let text = content_to_string(&msg.content);
                if !text.is_empty() {
                    blocks.push(ContentBlock::Text { text });
                }
                for tc in tool_calls {
                    let input: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Object(
                            serde_json::Map::new(),
                        ));
                    blocks.push(ContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        input,
                    });
                }
                MessageContent::Blocks(blocks)
            } else {
                let text = content_to_string(&msg.content);
                MessageContent::Text(text)
            }
        } else {
            let text = content_to_string(&msg.content);
            MessageContent::Text(text)
        };

        messages.push(AnthropicMessage {
            role: role.to_string(),
            content,
        });
    }

    let max_tokens = req.max_tokens.unwrap_or(4096);

    let anthropic_thinking = if thinking.enabled {
        Some(AnthropicThinking {
            thinking_type: "enabled".to_string(),
            budget_tokens: thinking.budget_tokens,
        })
    } else {
        None
    };

    MessagesRequest {
        model: remote_model.to_string(),
        messages,
        max_tokens,
        system,
        stream: req.stream,
        temperature: req.temperature,
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        thinking: anthropic_thinking,
        extra: Default::default(),
    }
}

// ── Anthropic → OpenAI response ──

pub fn anthropic_to_openai_response(resp: &MessagesResponse) -> ChatCompletionResponse {
    let mut content_text = String::new();
    let mut reasoning_text: Option<String> = None;
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in &resp.content {
        match block {
            ContentBlock::Text { text } => {
                content_text.push_str(text);
            }
            ContentBlock::Thinking { thinking, .. } => {
                let rt = reasoning_text.get_or_insert_with(String::new);
                rt.push_str(thinking);
            }
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                });
            }
            _ => {}
        }
    }

    let finish_reason = resp.stop_reason.as_deref().map(map_stop_reason);

    ChatCompletionResponse {
        id: resp.id.clone(),
        object: "chat.completion".to_string(),
        created: chrono_now_unix(),
        model: resp.model.clone(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: if content_text.is_empty() {
                    None
                } else {
                    Some(content_text)
                },
                reasoning_content: reasoning_text,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            finish_reason: finish_reason.map(|s| s.to_string()),
        }],
        usage: Some(Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        }),
        system_fingerprint: None,
    }
}

// ── Anthropic → OpenAI request (for /v1/messages → OpenAI backend) ──

pub fn anthropic_to_openai_request(
    req: &MessagesRequest,
    remote_model: &str,
) -> ChatCompletionRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    if let Some(system) = &req.system {
        let text = match system {
            SystemContent::Text(t) => t.clone(),
            SystemContent::Blocks(blocks) => {
                let texts: Vec<String> = blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()))
                    .collect();
                texts.join("\n")
            }
        };
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: Some(Value::String(text)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            extra: Default::default(),
        });
    }

    for msg in &req.messages {
        match &msg.content {
            MessageContent::Text(text) => {
                messages.push(ChatMessage {
                    role: msg.role.clone(),
                    content: Some(Value::String(text.clone())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                    extra: Default::default(),
                });
            }
            MessageContent::Blocks(blocks) => {
                let mut text_parts = Vec::new();
                let mut tool_calls_vec: Vec<ToolCall> = Vec::new();
                let mut tool_result_id: Option<String> = None;
                let mut tool_result_content: Option<String> = None;

                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            tool_calls_vec.push(ToolCall {
                                id: id.clone(),
                                call_type: "function".to_string(),
                                function: FunctionCall {
                                    name: name.clone(),
                                    arguments: serde_json::to_string(input).unwrap_or_default(),
                                },
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            tool_result_id = Some(tool_use_id.clone());
                            tool_result_content =
                                Some(content.as_str().unwrap_or("").to_string());
                        }
                        ContentBlock::Thinking { .. } => {}
                        _ => {}
                    }
                }

                if let Some(tc_id) = tool_result_id {
                    messages.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(Value::String(
                            tool_result_content.unwrap_or_default(),
                        )),
                        name: None,
                        tool_calls: None,
                        tool_call_id: Some(tc_id),
                        reasoning_content: None,
                        extra: Default::default(),
                    });
                } else {
                    messages.push(ChatMessage {
                        role: msg.role.clone(),
                        content: if text_parts.is_empty() {
                            None
                        } else {
                            Some(Value::String(text_parts.join("")))
                        },
                        name: None,
                        tool_calls: if tool_calls_vec.is_empty() {
                            None
                        } else {
                            Some(tool_calls_vec)
                        },
                        tool_call_id: None,
                        reasoning_content: None,
                        extra: Default::default(),
                    });
                }
            }
        }
    }

    ChatCompletionRequest {
        model: remote_model.to_string(),
        messages,
        stream: req.stream,
        stream_options: None,
        max_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        extra: Default::default(),
    }
}

// ── OpenAI → Anthropic response ──

pub fn openai_to_anthropic_response(resp: &ChatCompletionResponse) -> MessagesResponse {
    let mut content: Vec<ContentBlock> = Vec::new();

    if let Some(choice) = resp.choices.first() {
        if let Some(reasoning) = &choice.message.reasoning_content {
            content.push(ContentBlock::Thinking {
                thinking: reasoning.clone(),
                signature: None,
            });
        }
        if let Some(text) = &choice.message.content {
            content.push(ContentBlock::Text { text: text.clone() });
        }
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tc in tool_calls {
                let input: Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Object(
                        serde_json::Map::new(),
                    ));
                content.push(ContentBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    input,
                });
            }
        }
    }

    let stop_reason = resp
        .choices
        .first()
        .and_then(|c| c.finish_reason.as_deref())
        .map(map_finish_reason)
        .map(|s| s.to_string());

    let usage = resp.usage.as_ref().map_or(
        AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        },
        |u| AnthropicUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        },
    );

    MessagesResponse {
        id: resp.id.clone(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model.clone(),
        stop_reason,
        stop_sequence: None,
        usage,
    }
}

// ── Helpers ──

fn content_to_string(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| {
                if v.get("type").and_then(|t| t.as_str()) == Some("text") {
                    v.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn map_stop_reason(reason: &str) -> &str {
    match reason {
        "end_turn" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        "stop_sequence" => "stop",
        other => other,
    }
}

fn map_finish_reason(reason: &str) -> &str {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        other => other,
    }
}

fn chrono_now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

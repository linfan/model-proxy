#!/bin/bash

echo "=== OpenAI Chat Completions (stream) ==="
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "你是谁？用10个字以内回答"
        }
    ],
    "stream": true,
    "stream_options": {
        "include_usage": true
    }
}'

echo ""
echo ""
echo "=== OpenAI Chat Completions (non-stream) ==="
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "你是谁？用10个字以内回答"
        }
    ],
    "stream": false
}'

echo ""
echo ""
echo "=== Anthropic Messages (non-stream) ==="
curl -X POST http://127.0.0.1:8000/v1/messages \
-H "Content-Type: application/json" \
-d '{
    "model": "claude-sonnet",
    "max_tokens": 100,
    "messages": [
        {
            "role": "user",
            "content": "Who are you? Answer in 10 words or less."
        }
    ],
    "stream": false
}'

echo ""
echo ""
echo "=== Anthropic Messages with thinking (non-stream) ==="
curl -X POST http://127.0.0.1:8000/v1/messages \
-H "Content-Type: application/json" \
-d '{
    "model": "claude-sonnet",
    "max_tokens": 1000,
    "messages": [
        {
            "role": "user",
            "content": "What is 15 * 37?"
        }
    ],
    "thinking": {
        "type": "enabled",
        "budget_tokens": 5000
    },
    "stream": false
}'

echo ""
echo ""
echo "=== List Models ==="
curl http://127.0.0.1:8000/v1/models

echo ""

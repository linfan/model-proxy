# OpenAI Proxy

一个用于代理 OpenAI API 请求的服务，支持隐藏真实 API 密钥和记录请求响应。

## 环境变量配置

### CONFIG_FILE_PATH
设置配置文件路径，默认为 `${HOME}/.openai-proxy/model_mapping.json`

### RECORD_FILE_PATH
设置请求记录文件目录，默认为 `${HOME}/.openai-proxy/records`

### URL_PREFIX
通过设置 `URL_PREFIX` 环境变量可以为所有 API 添加公共前缀。

```bash
# 设置 URL 前缀为 /api
export URL_PREFIX=/api

# 启动服务
./dist/openai-proxy
```

设置后，原本的 API 路径会变为：
- `/v1/chat/completions` → `/api/v1/chat/completions`
- `/v1/models` → `/api/v1/models`

## 构建和运行

```bash
# 构建项目
make build

# 运行服务
./dist/openai-proxy

# 或者直接运行（带调试模式）
./dist/openai-proxy -d
```

## 使用示例

### 不使用前缀（默认）
```bash
curl http://localhost:8000/v1/models
```

### 使用前缀
```bash
export URL_PREFIX=/api
./dist/openai-proxy

# 请求时需要加上前缀
curl http://localhost:8000/api/v1/models
```

## Linux systemd 服务

如果需要在 Linux 服务器上作为系统服务运行，可以使用 systemd。

复制 systemd 服务文件：

```bash
sudo cp openai-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### 使用

```bash
# 启动服务
sudo systemctl start openai-proxy

# 设置开机自启
sudo systemctl enable openai-proxy

# 查看服务状态
sudo systemctl status openai-proxy

# 查看日志
sudo journalctl -u openai-proxy -f

# 重启服务
sudo systemctl restart openai-proxy

# 停止服务
sudo systemctl stop openai-proxy
```
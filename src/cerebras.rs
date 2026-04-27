// src/cerebras.rs
use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, GetTokenUsage, Usage,
    message::{AssistantContent, Message, UserContent},
};
use serde::{Deserialize, Serialize};

const CEREBRAS_BASE_URL: &str = "https://api.cerebras.ai/v1";

// ============ Client ============

#[derive(Clone)]
pub struct Client {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: CEREBRAS_BASE_URL.to_string(),
            http_client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self, std::env::VarError> {
        Ok(Self::new(std::env::var("CEREBRAS_API_KEY")?))
    }

    // Helper para criar agente diretamente (igual openai::Client::agent)
    pub fn agent(
        &self,
        model: impl Into<String>,
    ) -> rig::agent::AgentBuilder<CerebrasCompletionModel> {
        rig::agent::AgentBuilder::new(self.completion_model(model))
    }
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

// ============ CompletionClient Trait ============

impl CompletionClient for Client {
    type CompletionModel = CerebrasCompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        CerebrasCompletionModel {
            client: self.clone(),
            model: model.into(),
        }
    }
}

// ============ ProviderClient Trait ============

impl ProviderClient for Client {
    type Input = String;

    fn from_val(input: Self::Input) -> Self {
        Self::new(input)
    }

    fn from_env() -> Self {
        Self::from_env().expect("CEREBRAS_API_KEY not set")
    }
}

// ============ Completion Model ============

#[derive(Clone)]
pub struct CerebrasCompletionModel {
    client: Client,
    model: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CerebrasMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct CerebrasRequest {
    model: String,
    messages: Vec<CerebrasMessage>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CerebrasResponse {
    pub id: String,
    pub choices: Vec<CerebrasChoice>,
    pub usage: Option<CerebrasUsage>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CerebrasChoice {
    pub index: u32,
    pub message: CerebrasMessage,
    pub finish_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CerebrasUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// Implement GetTokenUsage for CerebrasResponse
impl GetTokenUsage for CerebrasResponse {
    fn token_usage(&self) -> Option<Usage> {
        self.usage.as_ref().map(|u| Usage {
            input_tokens: u.prompt_tokens as u64,
            output_tokens: u.completion_tokens as u64,
            total_tokens: u.total_tokens as u64,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
        })
    }
}

impl CompletionModel for CerebrasCompletionModel {
    type Response = CerebrasResponse;
    type StreamingResponse = CerebrasResponse;
    type Client = Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self {
            client: client.clone(),
            model: model.into(),
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let mut messages = Vec::new();

        // Preamble → system message
        if let Some(preamble) = request.preamble {
            messages.push(CerebrasMessage {
                role: "system".to_string(),
                content: preamble,
            });
        }

        // Chat history
        for msg in request.chat_history {
            match msg {
                Message::User { content, .. } => {
                    for part in content {
                        if let UserContent::Text(t) = part {
                            messages.push(CerebrasMessage {
                                role: "user".to_string(),
                                content: t.text,
                            });
                        }
                    }
                }
                Message::Assistant { content, .. } => {
                    for part in content {
                        if let AssistantContent::Text(t) = part {
                            messages.push(CerebrasMessage {
                                role: "assistant".to_string(),
                                content: t.text,
                            });
                        }
                    }
                }
                Message::System { content } => {
                    messages.push(CerebrasMessage {
                        role: "system".to_string(),
                        content,
                    });
                }
            }
        }

        let body = CerebrasRequest {
            model: self.model.clone(),
            messages,
        };

        let response = self
            .client
            .http_client
            .post(format!("{}/chat/completions", self.client.base_url))
            .header("Authorization", format!("Bearer {}", self.client.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                CompletionError::HttpError(rig::http_client::Error::Instance(Box::new(e)))
            })?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(CompletionError::HttpError(
                rig::http_client::Error::Instance(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("{}: {}", status, text),
                ))),
            ));
        }

        let data: CerebrasResponse = response.json().await.map_err(|e| {
            CompletionError::HttpError(rig::http_client::Error::Instance(Box::new(e)))
        })?;

        let content = data
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = data
            .usage
            .as_ref()
            .map(|u| Usage {
                input_tokens: u.prompt_tokens as u64,
                output_tokens: u.completion_tokens as u64,
                total_tokens: u.total_tokens as u64,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
            .unwrap_or(Usage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            });

        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text(content)),
            message_id: Some(data.id.clone()),
            usage,
            raw_response: data,
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<rig::streaming::StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
    {
        todo!("Streaming not yet implemented for Cerebras")
    }
}

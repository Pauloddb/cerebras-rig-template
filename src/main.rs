mod cerebras;

use rig::completion::Prompt;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenv::dotenv().ok();

    let client = cerebras::Client::from_env()?;

    // ✅ CORRETO: .agent() no client, .preamble() no AgentBuilder
    let agent = client
        .agent("llama3.1-8b")
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("Olá!").await?;
    println!("{response}");

    Ok(())
}

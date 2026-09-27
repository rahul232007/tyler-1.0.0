use std::process::{Command, Child};
use std::time::Duration;
use tokio::time::sleep;

#[tauri::command]
async fn check_system_status() -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    
    let backend_ok = match client.get("http://localhost:8000/health").send().await {
        Ok(res) => res.status().is_success(),
        Err(_) => false,
    };

    let ollama_ok = match client.get("http://localhost:11434/api/tags").send().await {
        Ok(res) => res.status().is_success(),
        Err(_) => false,
    };

    Ok(serde_json::json!({
        "backend_online": backend_ok,
        "ollama_online": ollama_ok,
        "status": if backend_ok { "ready" } else { "initializing" }
    }))
}

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![check_system_status])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

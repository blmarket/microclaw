#![cfg(unix)]

use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use microclaw::config::Config;
use microclaw::llm::create_provider;
use microclaw::llm_types::{Message, MessageContent, ResponseContentBlock};
use tokio::time::timeout;

fn env_lock() -> MutexGuard<'static, ()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct EnvVarGuard {
    key: String,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &str, value: String) -> Self {
        let previous = env::var(key).ok();
        env::set_var(key, value);
        Self {
            key: key.to_string(),
            previous,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.as_deref() {
            env::set_var(&self.key, previous);
        } else {
            env::remove_var(&self.key);
        }
    }
}

struct TempDirGuard {
    path: PathBuf,
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    env::temp_dir().join(format!("{prefix}-{}-{stamp}", std::process::id()))
}

fn install_fake_codex(dir: &Path) -> std::io::Result<()> {
    let script_path = dir.join("codex");
    fs::write(&script_path, fake_codex_script())?;
    let mut perms = fs::metadata(&script_path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&script_path, perms)?;
    Ok(())
}

fn fake_codex_script() -> &'static str {
    r#"#!/usr/bin/env node
const args = process.argv.slice(2);

if (args.length === 1 && args[0] === '--version') {
  process.stdout.write('codex-cli fake\n');
  process.exit(0);
}

if (args.length !== 3 || args[0] !== 'app-server' || args[1] !== '--listen' || args[2] !== 'stdio://') {
  process.stderr.write(`unexpected args: ${JSON.stringify(args)}\n`);
  process.exit(2);
}

const threadId = 'thread-test';
const turnId = 'turn-test';
const answerItemId = 'item-answer';
const authRefreshId = 91;
const finalMessage = JSON.stringify({
  content: [{ type: 'text', text: '4' }],
  stop_reason: 'end_turn',
});
let waitingForAuthRefresh = false;

function send(payload) {
  process.stdout.write(JSON.stringify(payload) + '\n');
}

function fail(id, message) {
  send({ id, error: { code: -32600, message } });
}

function handle(message) {
  const id = message.id;
  const method = message.method;
  const params = message.params || {};

  if (waitingForAuthRefresh && id === authRefreshId) {
    if (message.error) {
      process.stderr.write(`auth refresh request failed: ${JSON.stringify(message.error)}\n`);
      process.exit(3);
    }
    const result = message.result || {};
    if (result.accessToken !== 'test-access-token') {
      process.stderr.write(`unexpected access token from auth refresh: ${JSON.stringify(result)}\n`);
      process.exit(3);
    }
    if (result.chatgptAccountId !== 'acct-prev') {
      process.stderr.write(`unexpected account id from auth refresh: ${JSON.stringify(result)}\n`);
      process.exit(3);
    }
    if (result.chatgptPlanType !== null) {
      process.stderr.write(`unexpected plan type from auth refresh: ${JSON.stringify(result)}\n`);
      process.exit(3);
    }
    waitingForAuthRefresh = false;
    send({
      method: 'thread/tokenUsage/updated',
      params: {
        threadId,
        turnId,
        tokenUsage: {
          last: {
            inputTokens: 11,
            outputTokens: 3,
          },
        },
      },
    });
    send({
      method: 'item/completed',
      params: {
        threadId,
        turnId,
        item: {
          type: 'agentMessage',
          id: answerItemId,
          phase: 'final_answer',
          text: finalMessage,
        },
      },
    });
    send({
      method: 'turn/completed',
      params: {
        threadId,
        turn: {
          id: turnId,
          status: 'completed',
        },
      },
    });
    return;
  }

  if (method === 'initialize') {
    const capabilities = params.capabilities || {};
    if (capabilities.experimentalApi !== true) {
      fail(
        id,
        `expected initialize.capabilities.experimentalApi=true, got ${JSON.stringify(capabilities.experimentalApi)}`,
      );
      return;
    }
    send({
      id,
      result: {
        userAgent: 'fake-codex/0',
        platformFamily: 'unix',
        platformOs: 'linux',
      },
    });
    return;
  }

  if (method === 'thread/start') {
    const config = params.config || {};
    const tools = config.tools || {};
    if (Object.prototype.hasOwnProperty.call(tools, 'web_search') && tools.web_search === null) {
      fail(
        id,
        'failed to load configuration: data did not match any variant of untagged enum WebSearchToolConfigInput\nin `tools.web_search`\n',
      );
      return;
    }
    if (config.web_search !== 'disabled') {
      fail(id, `expected web_search=disabled, got ${JSON.stringify(config.web_search)}`);
      return;
    }
    if (params.persistExtendedHistory !== true) {
      fail(
        id,
        `expected persistExtendedHistory=true, got ${JSON.stringify(params.persistExtendedHistory)}`,
      );
      return;
    }
    if (tools.view_image !== false) {
      fail(id, `expected tools.view_image=false, got ${JSON.stringify(tools.view_image)}`);
      return;
    }
    send({ id, result: { thread: { id: threadId } } });
    return;
  }

  if (method === 'turn/start') {
    const input = Array.isArray(params.input) ? params.input : [];
    const text = input
      .filter((item) => item && item.type === 'text')
      .map((item) => item.text || '')
      .join('\n');
    if (!text.includes('what is 2+2?')) {
      fail(id, `turn/start payload did not include the user question: ${text}`);
      return;
    }
    const schema = params.outputSchema || {};
    const properties = schema.properties || {};
    if (!properties.content || !properties.stop_reason) {
      fail(id, 'turn/start did not include the expected output schema');
      return;
    }

    send({ id, result: { turn: { id: turnId } } });
    send({
      id: authRefreshId,
      method: 'account/chatgptAuthTokens/refresh',
      params: {
        reason: 'startup',
        previousAccountId: 'acct-prev',
      },
    });
    waitingForAuthRefresh = true;
    return;
  }

  fail(id, `unsupported method: ${method}`);
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  while (true) {
    const newline = buffer.indexOf('\n');
    if (newline === -1) {
      break;
    }
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) {
      continue;
    }
    handle(JSON.parse(line));
  }
});
process.stdin.resume();
"#
}

#[tokio::test]
async fn codex_app_provider_answers_simple_question_via_fake_app_server() {
    let _guard = env_lock();
    let fake_bin_dir = unique_temp_dir("microclaw-codex-app-test");
    let fake_codex_home = unique_temp_dir("microclaw-codex-home-test");
    fs::create_dir_all(&fake_bin_dir).unwrap();
    fs::create_dir_all(&fake_codex_home).unwrap();
    let _temp_dir = TempDirGuard {
        path: fake_bin_dir.clone(),
    };
    let _temp_home = TempDirGuard {
        path: fake_codex_home.clone(),
    };
    install_fake_codex(&fake_bin_dir).unwrap();

    let path_value = match env::var("PATH") {
        Ok(existing) if !existing.is_empty() => {
            format!("{}:{existing}", fake_bin_dir.display())
        }
        _ => fake_bin_dir.display().to_string(),
    };
    let _path_guard = EnvVarGuard::set("PATH", path_value);
    let _codex_home_guard = EnvVarGuard::set("CODEX_HOME", fake_codex_home.display().to_string());
    let _access_token_guard =
        EnvVarGuard::set("OPENAI_CODEX_ACCESS_TOKEN", "test-access-token".to_string());

    let config: Config = serde_yaml::from_str(
        r#"
llm_provider: codex-app
model: gpt-5.4
"#,
    )
    .unwrap();
    let provider = create_provider(&config);
    let response = provider
        .send_message(
            "",
            vec![Message {
                role: "user".into(),
                content: MessageContent::Text("what is 2+2?".into()),
            }],
            None,
        )
        .await
        .unwrap();

    assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(response.content.len(), 1);
    match &response.content[0] {
        ResponseContentBlock::Text { text } => assert_eq!(text, "4"),
        other => panic!("expected text response, got {other:?}"),
    }
    let usage = response.usage.expect("expected usage from token update");
    assert_eq!(usage.input_tokens, 11);
    assert_eq!(usage.output_tokens, 3);
}

#[tokio::test]
#[ignore = "requires live `codex app-server` access, Codex auth, and outbound network; run explicitly with `cargo test codex_app_provider_answers_simple_question_via_live_app_server --test codex_app_integration -- --ignored`"]
async fn codex_app_provider_answers_simple_question_via_live_app_server() {
    let config: Config = serde_yaml::from_str(
        r#"
llm_provider: codex-app
model: gpt-5.4
"#,
    )
    .unwrap();
    let provider = create_provider(&config);
    let response = timeout(
        Duration::from_secs(60),
        provider.send_message(
            "",
            vec![Message {
                role: "user".into(),
                content: MessageContent::Text("what is 2+2?".into()),
            }],
            None,
        ),
    )
    .await
    .expect("live codex app-server request timed out")
    .expect("live codex app-server request failed");

    assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));
    let text = response
        .content
        .iter()
        .filter_map(|block| match block {
            ResponseContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    dbg!(&text);
    assert!(
        text.contains('4'),
        "expected live codex-app response to contain `4`, got: {text:?}"
    );
}

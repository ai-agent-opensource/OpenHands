# Rewrite


## Goal
Transform excessive abstraction and complexity into concreteness and simplicity.
Or Find out why they are excessive abstraction into paper

## checking point
development speed

## Plan
[] need to make BaseAgent #core/agent.py
<!-- 에이전트가 어떤 행동을 할지, 어떤 속성을 가질지 등을 추상적으로 또는 구체적으로 구현할 수 있습니다. 예를 들어, 메시지 처리, 도구 사용, 상태 관리 등의 메서드를 포함할 수 있습니다 -->

### Code

#### runtime
```python
new_conversation() #manage_conversation.py from app.py

# -- deleted ---
create_new_conversation() #conversation_service.py
maybe_start_agent_loop() #_conversation_manager.py
initialize_agent() #session.py
start() #agent_session.py
_create_runtime() #agent_session.py
self.runtime.connect() -> DockerRuntime.connect() #docker_runtime.py
self.init_container() #docker_runtime.py
# --------------

DockerRuntime() # docker_runtime.py

# -- created --
BaseAgent() # /core/agent.py

AgentDirectExecutor()
create_agent() # conversation_service.py
# -------------

```

better structure
graph TD
    A[app.py - new_conversation] --> B[conversation_service.py - create_agent]
    B --> C[AgentDirectExecutor - run]
    C --> D[DockerRuntime - start]



### Memo
"conversation_store" need for tracking conversation status

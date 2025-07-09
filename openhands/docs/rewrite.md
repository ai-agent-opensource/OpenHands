## Rewrite


## Goal
Transform excessive abstraction and complexity into concreteness and simplicity.
Or Find out why they are excessive abstraction into paper

## Plan
[] rewrite docker_runtime.py in main.py

### Code

#### runtime
```python
new_conversation() #manage_conversation.py from app.py
create_new_conversation() #conversation_service.py
maybe_start_agent_loop() #_conversation_manager.py
initialize_agent() #session.py
start() #agent_session.py
_create_runtime() #agent_session.py
self.runtime.connect() -> DockerRuntime.connect() #docker_runtime.py
self.init_container() #docker_runtime.py
```

better structure
graph TD
    A[app.py - new_conversation] --> B[conversation_service.py - create_agent]
    B --> C[AgentDirectExecutor - run]
    C --> D[DockerRuntime - start]



### Memo
"conversation_store" need for tracking conversation status

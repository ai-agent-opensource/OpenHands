# Rewrite


## Goal
Transform excessive abstraction and complexity into concreteness and simplicity.
Or Find out why they are excessive abstraction into paper

## checking point
development speed

## TODO
[] fix Session() in #_conversation_manager.py


### Code

#### runtime
```python
new_conversation() #app.py.py

# -- deleted --- # Used place
create_new_conversation() #manage_conversation.py
    maybe_start_agent_loop() #conversation_service.py
        initialize_agent() #_conversation_manager.py
            MCPConfig() # session.py
            Agent.get_cls(agent_cls)(llm, agent_config) # session.py
            start() #session.py
                _create_runtime() # agent_session.py
                    self.runtime.connect() -> DockerRuntime.connect()
                            self.init_container() #docker_runtime.py
                _create_memory()
                _run_replay() or _create_controller()
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
"conversation_store" need for tracking conversationt
n status
- use Agent from session.py instead of making BaseAge
- agent_session.start() 안에서 구동시키지 말고, parameters에 넘겨서 사용해보면 어떨까? - fix current structure
```python
class AgentSession:
    def __init__(
        self,
        sid: str,
        runtime_factory: RuntimeFactory,
        memory_factory: MemoryFactory,
        controller_factory: ControllerFactory
    ):
        self.sid = sid
        self._runtime_factory = runtime_factory
        self._memory_factory = memory_factory
        self._controller_factory = controller_factory
```

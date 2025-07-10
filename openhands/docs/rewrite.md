# Rewrite


## Goal
Transform excessive abstraction and complexity into concreteness and simplicity.
Or Find out why they are excessive abstraction into paper

## checking point
development speed

## Plan
- [ ] **단계 0: 사전 준비 및 현황 파악**
    - [ ] 현행 구조(호출 흐름 및 각 함수의 책임) 명확히 파악
    - [ ] 기존 기능에 대한 단위 및 통합 테스트가 충분한지 확인 또는 보강

- [ ] **단계 1: 런타임 레이어 단순화 (`openhands/runtime/docker.py`)**
    - [ ] `DockerRuntime` 클래스 내 `connect()`와 `init_container()`의 기능 분석
    - [ ] 이 두 메서드의 핵심 로직을 포함하거나 호출하는 `async def start()` 메서드를 `DockerRuntime` 클래스 내에 구현
    - [ ] `connect()`와 `init_container()`를 외부에서 직접 호출하지 않도록 내부 메서드로 변경하거나 `start()` 메서드 내에서만 사용되도록 리팩토링
    - [ ] `DockerRuntime.start()`가 올바르게 작동하는지 확인하는 단위 테스트 작성 및 실행

- [ ] **단계 2: 에이전트 핵심 정의 (`openhands/core/agent.py` 생성)**
    - [ ] `openhands/core/agent.py` 파일 새로 생성
    - [ ] `BaseAgent` 클래스 정의 및 에이전트의 기본적인 속성과 추상 메서드(예: `process_message`, `execute_tool`) 포함
    - [ ] `BaseAgent` 인스턴스화 및 기본 메서드 호출에 대한 단위 테스트 작성

- [ ] **단계 3: 에이전트 실행 추상화 (`AgentDirectExecutor` 구현)**
    - [ ] `AgentDirectExecutor` 클래스를 적절한 경로(예: `openhands/core/agent_direct_executor.py`)에 구현
    - [ ] 생성자에서 `BaseAgent` 인스턴스와 `DockerRuntime` 인스턴스를 받도록 정의
    - [ ] `async def run()` 메서드 구현 (내부에 `self.runtime.start()` 및 `self.agent`의 주요 실행 로직 포함)
    - [ ] 예외 처리 및 로깅 포함
    - [ ] `AgentDirectExecutor`가 에이전트를 성공적으로 실행하는지 확인하는 통합 테스트 작성

- [ ] **단계 4: 대화 서비스 통합 (`openhands/server/services/conversation_service.py` 리팩토링)**
    - [ ] `async def create_agent(...)` 함수 새로 구현 (내부에 `BaseAgent` 및 `DockerRuntime` 인스턴스 생성, `AgentDirectExecutor` 호출)
    - [ ] 기존 `create_new_conversation()` 함수에서 `maybe_start_agent_loop()` 호출 부분을 `create_agent()` 호출로 대체 또는 래핑
    - [ ] 불필요해진 `maybe_start_agent_loop()`, `initialize_agent()`, `start()`, `_create_runtime()`, `self.runtime.connect()`, `self.init_container()`에 대한 직접적인 호출 제거
    - [ ] `create_agent()`를 통해 새로운 대화가 성공적으로 시작되고 에이전트가 올바르게 실행되는지 확인하는 통합 테스트 작성

- [ ] **단계 5: 최상위 진입점 업데이트 (`app.py` 또는 `manage_conversation.py`)**
    - [ ] `app.py` 또는 `manage_conversation.py` 파일 내 `new_conversation()` 함수 수정하여 `conversation_service.py`의 `create_agent()`를 직접 호출하도록 변경
    - [ ] 이전의 복잡한 대화 시작 로직 제거
    - [ ] 전체 애플리케이션 실행 및 새로운 대화 시작 흐름이 정상적으로 작동하는지 확인하는 엔드-투-엔드 테스트 수행

- [ ] **단계 6: 문서화 및 정리**
    - [ ] 리라이트된 부분에 대한 코드 주석 최신화
    - [ ] `rewrite.md` 파일을 새로운 구조와 변경된 호출 흐름을 반영하여 업데이트
    - [ ] 시스템 아키텍처 다이어그램(예: `better structure` 그래프) 최신화

---

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

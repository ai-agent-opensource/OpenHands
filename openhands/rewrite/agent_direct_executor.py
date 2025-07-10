# openhands/core/agent.py (예시 경로, 실제 에이전트 클래스 경로에 맞게 변경)
# from openhands.core.agent import BaseAgent as Agent # 에이전트의 기본 클래스를 임포트한다고 가정합니다.

# openhands/runtime/docker.py (예시 경로, 실제 런타임 클래스 경로에 맞게 변경)
# from openhands.runtime.docker import DockerRuntime # Docker 런타임 클래스를 임포트한다고 가정합니다.

# openhands/server/services/conversation_service.py 에서 이미 사용하고 있는 로거
from openhands.core.logger import openhands_logger as logger

class AgentDirectExecutor:
    """
    AgentDirectExecutor는 주어진 에이전트와 런타임 환경을 사용하여 에이전트의 실행을 직접 담당합니다.
    이 클래스는 에이전트의 주요 실행 로직을 시작하고 런타임을 관리합니다.
    """

    def __init__(self, agent_instance, runtime_instance):
        """
        AgentDirectExecutor를 초기화합니다.
        Args:
            agent_instance: 실행할 에이전트 인스턴스 (예: BaseAgent의 서브클래스 인스턴스)
            runtime_instance: 에이전트가 실행될 런타임 환경 인스턴스 (예: DockerRuntime 인스턴스)
        """
        self.agent = agent_instance
        self.runtime = runtime_instance
        logger.info(f"AgentDirectExecutor가 에이전트 ({type(agent_instance).__name__})와 런타임 ({type(runtime_instance).__name__})으로 초기화되었습니다.")

    async def run(self):
        """
        에이전트 실행을 시작하고 런타임을 연결합니다.
        에이전트의 핵심 실행 로직을 여기에 통합합니다.
        """
        logger.info("AgentDirectExecutor: 에이전트 실행을 시작합니다...")
        try:
            # 1. 런타임 환경 시작
            # 이 부분은 실제 DockerRuntime의 `start` 메서드 시그니처에 따라 조정되어야 합니다.
            # 일반적으로 런타임은 에이전트가 코드를 실행할 수 있는 환경을 준비합니다.
            await self.runtime.start()
            logger.info("AgentDirectExecutor: 런타임이 성공적으로 시작되었습니다.")

            # 2. 에이전트의 주요 실행 루프 시작
            # 여기에 에이전트의 핵심 로직을 호출하는 코드를 추가합니다.
            # 예를 들어, 에이전트가 메시지를 처리하고 응답을 생성하는 루프를 시작할 수 있습니다.
            # 이 메서드는 에이전트 구현에 따라 달라질 것입니다 (예: self.agent.run_loop(self.runtime)).
            # 예시:
            # await self.agent.run_with_runtime(self.runtime)
            logger.info("AgentDirectExecutor: 에이전트 실행 루프가 시작되었습니다.")

        except Exception as e:
            logger.error(f"AgentDirectExecutor: 에이전트 실행 중 오류 발생: {e}")
            raise # 오류를 다시 발생시켜 상위 호출자에게 알립니다.
        finally:
            # 3. 에이전트 실행이 완료된 후 런타임 정리 (필요하다면)
            # 예시:
            # if self.runtime.is_running:
            #    await self.runtime.stop()
            logger.info("AgentDirectExecutor: 에이전트 실행이 완료되었습니다.")
